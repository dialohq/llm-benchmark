use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use bytes::Bytes;
use http_body_util::{BodyExt, Empty};
use hyper::Request;
use hyper_util::client::legacy::{connect::HttpConnector, Client};
use hyper_util::rt::TokioExecutor;
use tokio::process::Child;

const POLL_INTERVAL: Duration = Duration::from_secs(2);

/// Poll http://127.0.0.1:<port>/v1/models until 200, the deadline
/// elapses, or the child exits.
pub async fn wait_until_ready(child: &mut Child, port: u16, timeout: Duration) -> Result<()> {
    let url = format!("http://127.0.0.1:{port}/v1/models");
    let deadline = Instant::now() + timeout;
    let client: Client<HttpConnector, Empty<Bytes>> =
        Client::builder(TokioExecutor::new()).build_http();

    loop {
        if let Some(status) = child.try_wait()? {
            return Err(anyhow!("child exited before becoming ready: {status}"));
        }
        if Instant::now() >= deadline {
            return Err(anyhow!(
                "engine ready timeout after {timeout:?} polling {url}"
            ));
        }
        match probe(&client, &url).await {
            Ok(true) => {
                tracing::info!("engine is ready ({url} → 200)");
                return Ok(());
            }
            Ok(false) => {}
            Err(e) => tracing::debug!("readiness probe error: {e:#}"),
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

async fn probe(client: &Client<HttpConnector, Empty<Bytes>>, url: &str) -> Result<bool> {
    let req = Request::builder()
        .method("GET")
        .uri(url)
        .body(Empty::<Bytes>::new())?;
    let resp = client.request(req).await?;
    let status = resp.status();
    let _ = resp.into_body().collect().await;
    Ok(status.is_success())
}
