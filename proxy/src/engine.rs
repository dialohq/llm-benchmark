use std::path::Path;
use std::process::Stdio;

use anyhow::{Context, Result};
use tokio::net::TcpListener;
use tokio::process::{Child, Command};

use crate::config::ResolvedConfig;

/// Bind 127.0.0.1:0, read the kernel-assigned port, drop the listener.
/// Tiny race window before the engine binds; for a local benchmark, fine.
pub async fn pick_loopback_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("binding 127.0.0.1:0 to pick a random port")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

pub fn spawn(cfg: &ResolvedConfig, repo_root: &Path, child_port: u16) -> Result<Child> {
    let mut argv = Vec::with_capacity(cfg.args.len() + 4);
    argv.extend(cfg.args.iter().cloned());
    argv.push("--host".into());
    argv.push("127.0.0.1".into());
    argv.push("--port".into());
    argv.push(child_port.to_string());

    let cwd = match &cfg.cwd {
        Some(p) if p.is_absolute() => p.clone(),
        Some(p) => repo_root.join(p),
        None => repo_root.to_path_buf(),
    };

    tracing::info!(
        cmd = %cfg.cmd,
        cwd = %cwd.display(),
        port = child_port,
        engine = cfg.engine.as_str(),
        "spawning child"
    );
    for (k, v) in &cfg.child_env {
        if cfg.secret_keys.contains(k) {
            tracing::info!("  env: {k}=<redacted, sha256={}>", short_hash(v));
        } else {
            tracing::info!("  env: {k}={v}");
        }
    }
    tracing::info!("  argv: {} {}", cfg.cmd, argv.join(" "));

    // Resolve cfg.cmd against the YAML's env.PATH (not the parent process's
    // PATH). Without this, `cmd: vllm` only works if the launcher already
    // happens to have the engine's venv/bin on its PATH — which the devshells
    // intentionally don't, so the YAML's env.PATH stops being authoritative.
    let cmd_program: std::path::PathBuf = {
        let raw = Path::new(&cfg.cmd);
        if raw.is_absolute() {
            raw.to_path_buf()
        } else if cfg.cmd.contains('/') {
            cwd.join(raw)
        } else if let Some(path) = cfg.child_env.get("PATH") {
            path.split(':')
                .filter(|p| !p.is_empty())
                .map(|dir| Path::new(dir).join(&cfg.cmd))
                .find(|c| c.is_file())
                .unwrap_or_else(|| raw.to_path_buf())
        } else {
            raw.to_path_buf()
        }
    };

    let mut cmd = Command::new(&cmd_program);
    cmd.args(&argv)
        .current_dir(&cwd)
        .env_clear()
        .envs(cfg.child_env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .stdin(Stdio::null())
        .kill_on_drop(true);

    let child = cmd.spawn().with_context(|| {
        format!(
            "spawning child `{}` (resolved={}, PATH from YAML env: {:?})",
            cfg.cmd,
            cmd_program.display(),
            cfg.child_env.get("PATH")
        )
    })?;

    Ok(child)
}

fn short_hash(v: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(v.as_bytes());
    hex::encode(&h.finalize()[..8])
}
