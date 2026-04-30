mod config;
mod engine;
mod meta;
mod proxy;
mod ready;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use bytes::Bytes;
use clap::Parser;
use tokio::signal::unix::{signal, SignalKind};

#[derive(Parser, Debug)]
#[command(version, about = "Inference-engine supervisor + reverse proxy.")]
struct Cli {
    /// Address to listen on for inbound /v1/* and /meta traffic.
    #[arg(long)]
    listen: SocketAddr,

    /// Path to the YAML launch config.
    #[arg(long)]
    config: PathBuf,

    /// Repository root used as the base for relative `cwd:` and `venv:`.
    /// Defaults to the directory containing the YAML.
    #[arg(long)]
    repo_root: Option<PathBuf>,

    /// Engine readiness timeout (seconds). Matches the smoke scripts' 1200s.
    #[arg(long, default_value_t = 1200)]
    ready_timeout_secs: u64,
}

fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(run(cli))
}

fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,bench_proxy=info"));
    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();
}

async fn run(cli: Cli) -> Result<()> {
    let config_abs = cli
        .config
        .canonicalize()
        .with_context(|| format!("resolving config path {}", cli.config.display()))?;
    let repo_root: PathBuf = match cli.repo_root {
        Some(p) => p.canonicalize().with_context(|| format!("--repo-root {}", p.display()))?,
        None => std::env::current_dir().context("getting cwd for default repo_root")?,
    };
    tracing::info!("repo_root: {}", repo_root.display());
    tracing::info!("config:    {}", config_abs.display());

    // Make ${REPO_ROOT} available to YAML env-value interpolation so configs
    // are portable across machines.
    std::env::set_var("REPO_ROOT", &repo_root);

    let cfg = config::load(&config_abs)?;
    tracing::info!("description: {}", cfg.description);
    tracing::info!("engine:      {}", cfg.engine.as_str());

    let child_port = engine::pick_loopback_port().await?;
    tracing::info!("child loopback port: {child_port}");

    let state = Arc::new(proxy::State::new(child_port));

    // Serve the proxy port early — /meta returns 503 until we publish.
    let serve_state = state.clone();
    let serve_addr = cli.listen;
    let serve_task = tokio::spawn(async move {
        if let Err(e) = proxy::serve(serve_addr, serve_state).await {
            tracing::error!("proxy server error: {e:#}");
        }
    });

    // Spawn engine.
    let mut child = engine::spawn(&cfg, &repo_root, child_port)?;
    let child_pid = child.id().context("child has no PID immediately after spawn")? as i32;
    tracing::info!("child PID: {child_pid}");

    // Static collection in parallel with engine warm-up. Futures are
    // built with owned values so they're 'static and easy to join.
    let venv_abs: Option<PathBuf> = cfg.venv.as_ref().map(|p| {
        if p.is_absolute() {
            p.clone()
        } else {
            repo_root.join(p)
        }
    });
    let host_fut = meta::collect_host();
    let gpus_fut = meta::collect_gpus();
    let lshw_fut = meta::collect_lshw();
    let venv_fut = meta::collect_venv(repo_root.clone(), cfg.venv.clone());
    let engine_version_fut = meta::engine_version(cfg.engine, venv_abs);

    // Concurrently: ready-poll the child.
    let ready_timeout = Duration::from_secs(cli.ready_timeout_secs);

    tokio::select! {
        // Bail if the child dies before becoming ready or during static
        // collection — we'd rather error out than serve a half-alive setup.
        res = orchestrate(
            &mut child,
            child_pid,
            child_port,
            cfg,
            cli.listen,
            state.clone(),
            ready_timeout,
            host_fut,
            gpus_fut,
            lshw_fut,
            venv_fut,
            engine_version_fut,
            repo_root,
        ) => {
            match res {
                Ok(()) => tracing::info!("snapshot published; serving traffic"),
                Err(e) => {
                    let _ = terminate_child(&mut child).await;
                    return Err(e);
                }
            }
        }
        _ = wait_termination_signal() => {
            tracing::info!("signal received during startup; aborting");
            let _ = terminate_child(&mut child).await;
            serve_task.abort();
            return Ok(());
        }
    }

    // Now wait for either: child exits (we propagate exit code) OR signal.
    tokio::select! {
        status = child.wait() => {
            let status = status?;
            tracing::info!("child exited: {status}");
            serve_task.abort();
            std::process::exit(status.code().unwrap_or(1));
        }
        _ = wait_termination_signal() => {
            tracing::info!("signal received; terminating child");
            terminate_child(&mut child).await;
            serve_task.abort();
            Ok(())
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn orchestrate(
    child: &mut tokio::process::Child,
    child_pid: i32,
    child_port: u16,
    cfg: config::ResolvedConfig,
    listen_addr: SocketAddr,
    state: Arc<proxy::State>,
    ready_timeout: Duration,
    host_fut: impl std::future::Future<Output = meta::HostSection>,
    gpus_fut: impl std::future::Future<Output = meta::GpusSection>,
    lshw_fut: impl std::future::Future<Output = serde_json::Value>,
    venv_fut: impl std::future::Future<Output = Option<meta::VenvSnapshot>>,
    engine_version_fut: impl std::future::Future<Output = Option<String>>,
    repo_root: PathBuf,
) -> Result<()> {
    let _ = listen_addr; // currently informational; reserved for future use
    let _ = repo_root;

    // Run static collection & ready-poll concurrently.
    let static_collect = async {
        let (host, gpus, lshw, venv, engine_version) =
            tokio::join!(host_fut, gpus_fut, lshw_fut, venv_fut, engine_version_fut);
        (host, gpus, lshw, venv, engine_version)
    };
    let ready = ready::wait_until_ready(child, child_port, ready_timeout);

    let ((host, gpus, lshw, venv, engine_version), ready_res) =
        tokio::join!(static_collect, ready);
    ready_res?;

    // Warmup *after* ready, *before* /proc/maps snapshot — so lazy CUDA
    // dlopens are present.
    let warmup = meta::run_warmup(child_port, cfg.warmup).await;

    // /proc snapshot (process tree, env from environ, loaded libs).
    let proc_snap = tokio::task::spawn_blocking({
        let secret_keys = cfg.secret_keys.clone();
        move || meta::collect_proc(child_pid, &secret_keys)
    })
    .await
    .context("join /proc collector")??;

    let snapshot = meta::Snapshot {
        schema_version: 1,
        started_at: chrono::Utc::now().to_rfc3339(),
        config: meta::config_section(&cfg),
        engine: cfg.engine.as_str(),
        engine_version,
        command: proc_snap.command,
        child_pid: proc_snap.child_pid,
        child_port,
        warmup,
        host,
        gpus,
        lshw,
        venv_snapshot: venv,
        child_runtime: proc_snap.child_runtime,
    };

    let json = serde_json::to_vec(&snapshot).context("serialize snapshot")?;
    state.publish_meta(Bytes::from(json)).await;
    Ok(())
}

async fn wait_termination_signal() {
    // SIGINT or SIGTERM, whichever first.
    let mut sigint = match signal(SignalKind::interrupt()) {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut sigterm = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(_) => return,
    };
    tokio::select! {
        _ = sigint.recv() => {},
        _ = sigterm.recv() => {},
    }
}

/// SIGTERM with a grace period, then SIGKILL via tokio's start_kill.
async fn terminate_child(child: &mut tokio::process::Child) {
    let pid = match child.id() {
        Some(p) => p as i32,
        None => return,
    };
    unsafe {
        libc::kill(pid, libc::SIGTERM);
    }
    tokio::select! {
        res = child.wait() => {
            match res {
                Ok(s) => tracing::info!("child exited after SIGTERM: {s}"),
                Err(e) => tracing::warn!("waiting on child after SIGTERM: {e:#}"),
            }
        }
        _ = tokio::time::sleep(Duration::from_secs(15)) => {
            tracing::warn!("child didn't exit after 15s; SIGKILL");
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
    }
}
