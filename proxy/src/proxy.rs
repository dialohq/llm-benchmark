//! Reverse proxy. Hot path: byte-shovel SSE chunks from upstream to
//! downstream with no buffering, no copy-into-vec, no compression.
//! Anything fancier than `boxed()` here is a chance to inflate TTFT.

use std::convert::Infallible;
use std::error::Error as StdError;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use bytes::Bytes;
use http_body_util::{combinators::BoxBody, BodyExt, Full};
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{header, Method, Request, Response, StatusCode, Uri};
use hyper_util::client::legacy::{connect::HttpConnector, Client};
use hyper_util::rt::{TokioExecutor, TokioIo};
use tokio::net::TcpListener;
use tokio::sync::RwLock;

type BoxError = Box<dyn StdError + Send + Sync>;
type ProxyBody = BoxBody<Bytes, BoxError>;

pub struct State {
    pub target_port: u16,
    pub meta: RwLock<Option<Bytes>>,
    client: Client<HttpConnector, ProxyBody>,
}

impl State {
    pub fn new(target_port: u16) -> Self {
        let mut connector = HttpConnector::new();
        connector.set_nodelay(true);
        let client = Client::builder(TokioExecutor::new())
            .pool_idle_timeout(Some(Duration::from_secs(30)))
            .build(connector);
        Self {
            target_port,
            meta: RwLock::new(None),
            client,
        }
    }

    pub async fn publish_meta(&self, json: Bytes) {
        *self.meta.write().await = Some(json);
    }
}

pub async fn serve(addr: SocketAddr, state: Arc<State>) -> Result<()> {
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("listening on {addr}");
    loop {
        let (stream, peer) = listener.accept().await?;
        let _ = stream.set_nodelay(true);
        let state = state.clone();
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let svc = service_fn(move |req| {
                let state = state.clone();
                async move { handle(req, state).await }
            });
            if let Err(e) = http1::Builder::new()
                .keep_alive(true)
                .serve_connection(io, svc)
                .with_upgrades()
                .await
            {
                tracing::debug!("conn from {peer}: {e:#}");
            }
        });
    }
}

async fn handle(
    req: Request<Incoming>,
    state: Arc<State>,
) -> Result<Response<ProxyBody>, Infallible> {
    let path = req.uri().path();
    if req.method() == Method::GET && path == "/meta" {
        return Ok(meta_response(&state).await);
    }
    if req.method() == Method::GET && path == "/healthz" {
        return Ok(simple_response(StatusCode::OK, "ok"));
    }
    Ok(proxy(req, &state).await)
}

async fn meta_response(state: &State) -> Response<ProxyBody> {
    let g = state.meta.read().await;
    match &*g {
        None => {
            let mut resp =
                simple_response(StatusCode::SERVICE_UNAVAILABLE, "meta not ready\n");
            resp.headers_mut()
                .insert(header::RETRY_AFTER, header::HeaderValue::from_static("5"));
            resp
        }
        Some(bytes) => {
            let body = Full::new(bytes.clone())
                .map_err(|never: Infallible| match never {})
                .boxed();
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/json")
                .body(body)
                .expect("response build")
        }
    }
}

async fn proxy(req: Request<Incoming>, state: &State) -> Response<ProxyBody> {
    if state.meta.read().await.is_none() {
        let mut resp = simple_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "engine not ready (warmup in progress)\n",
        );
        resp.headers_mut()
            .insert(header::RETRY_AFTER, header::HeaderValue::from_static("5"));
        return resp;
    }

    let target = format!("127.0.0.1:{}", state.target_port);
    let path_query = req
        .uri()
        .path_and_query()
        .map(|x| x.as_str())
        .unwrap_or("/");
    let upstream_uri: Uri = match format!("http://{target}{path_query}").parse() {
        Ok(u) => u,
        Err(e) => return simple_response(StatusCode::BAD_REQUEST, &format!("bad uri: {e}\n")),
    };

    let (mut parts, body) = req.into_parts();
    parts.uri = upstream_uri;
    strip_hop_by_hop(&mut parts.headers);
    if let Ok(host_v) = header::HeaderValue::from_str(&target) {
        parts.headers.insert(header::HOST, host_v);
    }
    let upstream_body = body.map_err(|e| Box::new(e) as BoxError).boxed();
    let upstream_req = Request::from_parts(parts, upstream_body);

    match state.client.request(upstream_req).await {
        Ok(resp) => {
            let (mut parts, body) = resp.into_parts();
            strip_hop_by_hop(&mut parts.headers);
            let body = body.map_err(|e| Box::new(e) as BoxError).boxed();
            Response::from_parts(parts, body)
        }
        Err(e) => {
            tracing::warn!("upstream error: {e:#}");
            simple_response(
                StatusCode::BAD_GATEWAY,
                &format!("upstream error: {e}\n"),
            )
        }
    }
}

fn strip_hop_by_hop(headers: &mut hyper::HeaderMap) {
    for h in [
        header::CONNECTION,
        header::TRANSFER_ENCODING,
        header::UPGRADE,
        header::PROXY_AUTHENTICATE,
        header::PROXY_AUTHORIZATION,
        header::TE,
        header::TRAILER,
    ] {
        headers.remove(h);
    }
    headers.remove("keep-alive");
    headers.remove("proxy-connection");
}

fn simple_response(status: StatusCode, msg: &str) -> Response<ProxyBody> {
    let body = Full::new(Bytes::from(msg.to_string()))
        .map_err(|never: Infallible| match never {})
        .boxed();
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(body)
        .expect("simple response build")
}
