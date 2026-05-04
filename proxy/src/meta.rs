//! Snapshot collection. The whole point of the proxy: capture every detail
//! that could affect benchmark numbers, frozen at startup after warmup.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use http_body_util::{BodyExt, Empty, Full};
use hyper::Request;
use hyper_util::client::legacy::{connect::HttpConnector, Client};
use hyper_util::rt::TokioExecutor;
use indexmap::IndexMap;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::process::Command;

use crate::config::{Engine, ResolvedConfig};

// ---------------------------------------------------------------------------
// Snapshot schema (serialized as the /meta JSON body).
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct Snapshot {
    pub schema_version: u32,
    pub started_at: String,
    pub config: ConfigSection,
    pub engine: &'static str,
    pub engine_version: Option<String>,
    pub command: CommandSection,
    pub child_pid: i32,
    pub child_port: u16,
    pub warmup: WarmupResult,
    pub host: HostSection,
    pub gpus: GpusSection,
    pub lshw: Value,
    pub venv_snapshot: Option<VenvSnapshot>,
    pub child_runtime: ChildRuntime,
}

#[derive(Debug, Serialize)]
pub struct ConfigSection {
    pub path: PathBuf,
    pub sha256: String,
    pub raw: String,
    pub description: String,
}

#[derive(Debug, Serialize)]
pub struct CommandSection {
    pub argv: Vec<String>,
    pub cwd: PathBuf,
    pub env: IndexMap<String, EnvEntry>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum EnvEntry {
    Plain(String),
    Redacted {
        redacted: bool,
        sha256: String,
    },
}

impl EnvEntry {
    fn build(value: String, is_secret: bool) -> Self {
        if is_secret {
            let mut h = Sha256::new();
            h.update(value.as_bytes());
            EnvEntry::Redacted {
                redacted: true,
                sha256: hex::encode(h.finalize()),
            }
        } else {
            EnvEntry::Plain(value)
        }
    }
}

#[derive(Debug, Serialize)]
pub struct WarmupResult {
    pub requested: u32,
    pub completed: u32,
    pub elapsed_ms: Vec<u64>,
}

#[derive(Debug, Serialize)]
pub struct HostSection {
    pub hostname: String,
    pub kernel: String,
    pub os_release: BTreeMap<String, String>,
    pub cpu: Value,
    pub memory_kb: u64,
    pub tunables: HostTunables,
}

#[derive(Debug, Serialize)]
pub struct HostTunables {
    pub thp_enabled: Option<String>,
    pub thp_defrag: Option<String>,
    pub cpu_governor: Vec<String>,
    pub numa: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct GpusSection {
    pub driver_version: Option<String>,
    pub cuda_driver_version: Option<String>,
    pub summary: Vec<Value>,
    pub nvidia_smi_xml: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct VenvSnapshot {
    pub root: PathBuf,
    pub python_version: Option<String>,
    pub python_realpath: Option<PathBuf>,
    pub uv_lock_path: Option<PathBuf>,
    pub uv_lock_sha256: Option<String>,
    pub uv_pip_freeze: Option<Vec<String>>,
    pub file_count: u64,
    pub total_bytes: u64,
    pub merkle_sha256: String,
    pub files: Vec<VenvFile>,
}

#[derive(Debug, Serialize)]
pub struct VenvFile {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Serialize)]
pub struct ChildRuntime {
    pub proc_status: String,
    pub proc_limits: String,
    pub process_tree: Vec<ProcessEntry>,
    pub loaded_libs: Vec<LoadedLib>,
}

#[derive(Debug, Serialize)]
pub struct ProcessEntry {
    pub pid: i32,
    pub ppid: i32,
    pub comm: String,
    pub cmdline: Vec<String>,
    pub env: IndexMap<String, EnvEntry>,
}

#[derive(Debug, Serialize)]
pub struct LoadedLib {
    pub path: PathBuf,
    pub version: Option<String>,
    pub inode: u64,
    pub size_bytes: u64,
    pub sha256: String,
    pub loaded_in_pids: Vec<i32>,
}

// ---------------------------------------------------------------------------
// Collection — host
// ---------------------------------------------------------------------------

pub async fn collect_host() -> HostSection {
    let hostname = run_capture("hostname", &[])
        .await
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let kernel = run_capture("uname", &["-srvm"])
        .await
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let os_release = parse_os_release()
        .await
        .unwrap_or_default();
    let cpu = match run_capture("lscpu", &["-J"]).await {
        Ok(s) => serde_json::from_str(&s).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let memory_kb = parse_meminfo_total().await.unwrap_or(0);
    let tunables = collect_tunables().await;
    HostSection {
        hostname,
        kernel,
        os_release,
        cpu,
        memory_kb,
        tunables,
    }
}

async fn parse_os_release() -> Result<BTreeMap<String, String>> {
    let s = tokio::fs::read_to_string("/etc/os-release").await?;
    let mut out = BTreeMap::new();
    for line in s.lines() {
        if let Some((k, v)) = line.split_once('=') {
            let v = v.trim().trim_matches('"').to_string();
            out.insert(k.trim().to_string(), v);
        }
    }
    Ok(out)
}

async fn parse_meminfo_total() -> Result<u64> {
    let s = tokio::fs::read_to_string("/proc/meminfo").await?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb = rest
                .trim()
                .split_whitespace()
                .next()
                .and_then(|n| n.parse::<u64>().ok())
                .ok_or_else(|| anyhow!("can't parse MemTotal: {line:?}"))?;
            return Ok(kb);
        }
    }
    Err(anyhow!("MemTotal not found in /proc/meminfo"))
}

async fn collect_tunables() -> HostTunables {
    let thp_enabled = tokio::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled")
        .await
        .ok()
        .map(|s| s.trim().to_string());
    let thp_defrag = tokio::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/defrag")
        .await
        .ok()
        .map(|s| s.trim().to_string());
    let mut cpu_governor = Vec::new();
    if let Ok(mut entries) = tokio::fs::read_dir("/sys/devices/system/cpu").await {
        let mut buf: Vec<(usize, String)> = Vec::new();
        while let Ok(Some(e)) = entries.next_entry().await {
            let name = e.file_name().to_string_lossy().to_string();
            if let Some(rest) = name.strip_prefix("cpu") {
                if let Ok(n) = rest.parse::<usize>() {
                    let p = e.path().join("cpufreq/scaling_governor");
                    if let Ok(s) = tokio::fs::read_to_string(&p).await {
                        buf.push((n, s.trim().to_string()));
                    }
                }
            }
        }
        buf.sort_by_key(|(n, _)| *n);
        cpu_governor = buf.into_iter().map(|(_, g)| g).collect();
    }
    let numa = run_capture("numactl", &["--hardware"]).await.ok();
    HostTunables {
        thp_enabled,
        thp_defrag,
        cpu_governor,
        numa,
    }
}

// ---------------------------------------------------------------------------
// Collection — gpus
// ---------------------------------------------------------------------------

pub async fn collect_gpus() -> GpusSection {
    let driver_version = nvidia_smi_field("driver_version").await;
    let cuda_driver_version = nvidia_smi_field("cuda_version").await;

    let summary = match run_capture(
        "nvidia-smi",
        &[
            "--query-gpu=index,name,uuid,memory.total,compute_cap,pstate,persistence_mode,power.management,clocks.applications.graphics,clocks.applications.memory",
            "--format=csv,noheader",
        ],
    )
    .await
    {
        Ok(csv) => parse_gpu_csv(&csv),
        Err(_) => Vec::new(),
    };

    let nvidia_smi_xml = run_capture("nvidia-smi", &["-q", "-x"]).await.ok();

    GpusSection {
        driver_version,
        cuda_driver_version,
        summary,
        nvidia_smi_xml,
    }
}

async fn nvidia_smi_field(field: &str) -> Option<String> {
    let out = run_capture(
        "nvidia-smi",
        &[
            &format!("--query-gpu={field}"),
            "--format=csv,noheader",
        ],
    )
    .await
    .ok()?;
    out.lines().next().map(|s| s.trim().to_string())
}

fn parse_gpu_csv(csv: &str) -> Vec<Value> {
    let headers = [
        "index",
        "name",
        "uuid",
        "memory_total_mib",
        "compute_cap",
        "pstate",
        "persistence_mode",
        "power_management",
        "clocks_app_graphics_mhz",
        "clocks_app_memory_mhz",
    ];
    let mut out = Vec::new();
    for line in csv.lines() {
        let cells: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if cells.len() != headers.len() {
            continue;
        }
        let mut obj = serde_json::Map::new();
        for (h, v) in headers.iter().zip(cells.iter()) {
            obj.insert((*h).into(), Value::String((*v).to_string()));
        }
        out.push(Value::Object(obj));
    }
    out
}

pub async fn collect_lshw() -> Value {
    match run_capture(
        "lshw",
        &[
            "-json",
            "-quiet",
            "-class", "system",
            "-class", "processor",
            "-class", "memory",
            "-class", "display",
            "-class", "network",
        ],
    )
    .await
    {
        Ok(s) => serde_json::from_str(&s).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    }
}

// ---------------------------------------------------------------------------
// Collection — venv
// ---------------------------------------------------------------------------

/// Walk `<venv>/{bin,lib,lib64}` (skipping __pycache__), sha256 every file,
/// emit a merkle hash for one-line-diffability. CPU-bound — runs in a
/// blocking pool.
pub async fn collect_venv(repo_root: PathBuf, venv_rel: Option<PathBuf>) -> Option<VenvSnapshot> {
    let venv = venv_rel?;
    let venv = if venv.is_absolute() {
        venv
    } else {
        repo_root.join(venv)
    };
    if !venv.is_dir() {
        tracing::warn!(
            "venv path {} does not exist; skipping venv_snapshot",
            venv.display()
        );
        return None;
    }
    let venv_path = venv.clone();
    let walk = tokio::task::spawn_blocking(move || walk_venv_files(&venv_path)).await;
    let mut files = match walk {
        Ok(Ok(f)) => f,
        Ok(Err(e)) => {
            tracing::warn!("venv walk failed: {e:#}");
            return None;
        }
        Err(e) => {
            tracing::warn!("venv walk join failed: {e:#}");
            return None;
        }
    };
    files.sort_by(|a, b| a.path.cmp(&b.path));

    let mut merkle = Sha256::new();
    let mut total_bytes = 0u64;
    for f in &files {
        merkle.update(f.path.as_os_str().as_encoded_bytes());
        merkle.update(b"\0");
        merkle.update(f.sha256.as_bytes());
        merkle.update(b"\n");
        total_bytes += f.size_bytes;
    }
    let merkle_sha256 = hex::encode(merkle.finalize());

    let python = venv.join("bin/python");
    let python_version = run_capture(python.to_string_lossy().as_ref(), &["-V"])
        .await
        .ok()
        .map(|s| s.trim().to_string());
    let python_realpath = tokio::fs::canonicalize(&python).await.ok();

    // uv pip freeze (best effort — works inside engine devshells)
    let uv_pip_freeze = run_capture(
        "uv",
        &[
            "pip",
            "freeze",
            "--python",
            python.to_string_lossy().as_ref(),
        ],
    )
    .await
    .ok()
    .map(|s| s.lines().map(|l| l.to_string()).collect::<Vec<_>>());

    // uv.lock alongside the venv — convention in this repo: <engine>/uv.lock
    let uv_lock_path = venv
        .parent()
        .map(|p| p.join("uv.lock"))
        .filter(|p| p.is_file());
    let uv_lock_sha256 = match &uv_lock_path {
        Some(p) => tokio::fs::read(p)
            .await
            .ok()
            .map(|b| hex::encode(Sha256::digest(&b))),
        None => None,
    };

    Some(VenvSnapshot {
        root: venv,
        python_version,
        python_realpath,
        uv_lock_path,
        uv_lock_sha256,
        uv_pip_freeze,
        file_count: files.len() as u64,
        total_bytes,
        merkle_sha256,
        files,
    })
}

fn walk_venv_files(venv: &Path) -> Result<Vec<VenvFile>> {
    use std::io::Read;
    let mut out = Vec::new();
    for sub in ["bin", "lib", "lib64"] {
        let root = venv.join(sub);
        if !root.exists() {
            continue;
        }
        for entry in walkdir::WalkDir::new(&root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| e.file_name() != "__pycache__")
        {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    tracing::debug!("walk: {e}");
                    continue;
                }
            };
            if !entry.file_type().is_file() {
                continue;
            }
            let abs = entry.path();
            let rel = match abs.strip_prefix(venv) {
                Ok(p) => p.to_path_buf(),
                Err(_) => abs.to_path_buf(),
            };
            let mut f = match std::fs::File::open(abs) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let mut hasher = Sha256::new();
            let mut buf = [0u8; 1 << 16];
            let mut size: u64 = 0;
            loop {
                let n = match f.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(_) => break,
                };
                hasher.update(&buf[..n]);
                size += n as u64;
            }
            out.push(VenvFile {
                path: rel,
                size_bytes: size,
                sha256: hex::encode(hasher.finalize()),
            });
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Collection — process tree, /proc reads, command, loaded libs
// ---------------------------------------------------------------------------

pub struct ProcSnapshot {
    pub command: CommandSection,
    pub child_pid: i32,
    pub child_runtime: ChildRuntime,
}

pub fn collect_proc(root_pid: i32, secret_keys: &HashSet<String>) -> Result<ProcSnapshot> {
    let tree = walk_descendants(root_pid)?;
    let mut process_tree = Vec::new();
    let mut pid_to_libs: HashMap<(u64, PathBuf), BTreeSet<i32>> = HashMap::new();

    for pid in &tree {
        let comm = read_to_string(&format!("/proc/{pid}/comm"))
            .unwrap_or_default()
            .trim()
            .to_string();
        let ppid = read_ppid(*pid).unwrap_or(0);
        let cmdline = read_cmdline(*pid).unwrap_or_default();
        let env = read_environ(*pid, secret_keys).unwrap_or_default();

        process_tree.push(ProcessEntry {
            pid: *pid,
            ppid,
            comm,
            cmdline,
            env,
        });

        if let Ok(libs) = read_maps(*pid) {
            for (inode, path) in libs {
                pid_to_libs.entry((inode, path)).or_default().insert(*pid);
            }
        }
    }

    let command_argv = read_cmdline(root_pid)?;
    let command_env = read_environ(root_pid, secret_keys)?;
    let command_cwd = std::fs::read_link(format!("/proc/{root_pid}/cwd"))
        .unwrap_or_else(|_| PathBuf::from("/"));

    let proc_status = read_to_string(&format!("/proc/{root_pid}/status")).unwrap_or_default();
    let proc_limits = read_to_string(&format!("/proc/{root_pid}/limits")).unwrap_or_default();

    let mut loaded_libs = Vec::with_capacity(pid_to_libs.len());
    for ((inode, path), pids) in pid_to_libs {
        let (size_bytes, sha256) = match hash_file(&path) {
            Ok((s, h)) => (s, h),
            Err(_) => (0, String::new()),
        };
        let version = parse_so_version(&path);
        loaded_libs.push(LoadedLib {
            path,
            version,
            inode,
            size_bytes,
            sha256,
            loaded_in_pids: pids.into_iter().collect(),
        });
    }
    loaded_libs.sort_by(|a, b| a.path.cmp(&b.path));

    Ok(ProcSnapshot {
        command: CommandSection {
            argv: command_argv,
            cwd: command_cwd,
            env: command_env,
        },
        child_pid: root_pid,
        child_runtime: ChildRuntime {
            proc_status,
            proc_limits,
            process_tree,
            loaded_libs,
        },
    })
}

fn read_to_string(p: &str) -> Result<String> {
    Ok(std::fs::read_to_string(p)?)
}

fn read_ppid(pid: i32) -> Result<i32> {
    let s = std::fs::read_to_string(format!("/proc/{pid}/status"))?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("PPid:") {
            return Ok(rest.trim().parse()?);
        }
    }
    Err(anyhow!("PPid not found in /proc/{pid}/status"))
}

fn read_cmdline(pid: i32) -> Result<Vec<String>> {
    let bytes = std::fs::read(format!("/proc/{pid}/cmdline"))?;
    Ok(bytes
        .split(|&b| b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| String::from_utf8_lossy(s).into_owned())
        .collect())
}

fn read_environ(pid: i32, secret_keys: &HashSet<String>) -> Result<IndexMap<String, EnvEntry>> {
    let bytes = std::fs::read(format!("/proc/{pid}/environ"))?;
    let mut out = IndexMap::new();
    for entry in bytes.split(|&b| b == 0).filter(|s| !s.is_empty()) {
        let s = String::from_utf8_lossy(entry);
        if let Some((k, v)) = s.split_once('=') {
            let is_secret = secret_keys.contains(k);
            out.insert(k.to_string(), EnvEntry::build(v.to_string(), is_secret));
        }
    }
    Ok(out)
}

/// Read all PIDs from /proc, build a child→parent map, and return all
/// descendants of `root` (inclusive of `root` itself).
fn walk_descendants(root: i32) -> Result<Vec<i32>> {
    let mut by_parent: HashMap<i32, Vec<i32>> = HashMap::new();
    for entry in std::fs::read_dir("/proc")? {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let name = entry.file_name();
        let name = match name.to_str() {
            Some(n) => n,
            None => continue,
        };
        let pid: i32 = match name.parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        if let Ok(ppid) = read_ppid(pid) {
            by_parent.entry(ppid).or_default().push(pid);
        }
    }
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(pid) = stack.pop() {
        out.push(pid);
        if let Some(children) = by_parent.get(&pid) {
            stack.extend_from_slice(children);
        }
    }
    out.sort();
    Ok(out)
}

fn read_maps(pid: i32) -> Result<Vec<(u64, PathBuf)>> {
    let s = std::fs::read_to_string(format!("/proc/{pid}/maps"))?;
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for line in s.lines() {
        // Format: addr-range perms offset dev inode pathname
        let mut it = line.split_ascii_whitespace();
        let _range = match it.next() {
            Some(x) => x,
            None => continue,
        };
        let _perms = it.next();
        let _offset = it.next();
        let _dev = it.next();
        let inode = match it.next().and_then(|x| x.parse::<u64>().ok()) {
            Some(n) => n,
            None => continue,
        };
        let path = match it.next() {
            Some(p) => p,
            None => continue,
        };
        // The pathname can contain spaces; join the rest.
        let mut pathbuf = path.to_string();
        for tok in it {
            pathbuf.push(' ');
            pathbuf.push_str(tok);
        }
        if inode == 0 || pathbuf.starts_with('[') || pathbuf.starts_with("anon_inode:") {
            continue;
        }
        if pathbuf.ends_with(" (deleted)") {
            continue;
        }
        let key = (inode, pathbuf.clone());
        if seen.insert(key.clone()) {
            out.push((inode, PathBuf::from(pathbuf)));
        }
    }
    Ok(out)
}

fn hash_file(path: &Path) -> Result<(u64, String)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 1 << 16];
    let mut size = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        size += n as u64;
    }
    Ok((size, hex::encode(h.finalize())))
}

/// Parse "libfoo.so.1.2.3" → "1.2.3", "libfoo.so" → None.
fn parse_so_version(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let idx = name.find(".so.")?;
    let rest = &name[idx + 4..];
    if rest.is_empty() {
        return None;
    }
    Some(rest.to_string())
}

// ---------------------------------------------------------------------------
// Warmup
// ---------------------------------------------------------------------------

pub async fn run_warmup(child_port: u16, count: u32) -> WarmupResult {
    if count == 0 {
        return WarmupResult {
            requested: 0,
            completed: 0,
            elapsed_ms: vec![],
        };
    }
    let model = match discover_model(child_port).await {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!("can't discover served model for warmup: {e:#}; skipping");
            return WarmupResult {
                requested: count,
                completed: 0,
                elapsed_ms: vec![],
            };
        }
    };

    let client: Client<HttpConnector, Full<Bytes>> =
        Client::builder(TokioExecutor::new()).build_http();

    let body_json = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Hello, this is a warmup."}],
        "max_tokens": 32,
        "temperature": 0,
        "stream": true,
    });
    let body_bytes = serde_json::to_vec(&body_json).expect("warmup body serializable");

    let url = format!("http://127.0.0.1:{child_port}/v1/chat/completions");
    let mut elapsed_ms = Vec::with_capacity(count as usize);
    let mut completed = 0u32;
    for i in 0..count {
        let start = std::time::Instant::now();
        let req = match Request::builder()
            .method("POST")
            .uri(&url)
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(body_bytes.clone())))
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("warmup #{i}: build request failed: {e:#}");
                break;
            }
        };
        let resp = match client.request(req).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("warmup #{i}: request failed: {e:#}");
                break;
            }
        };
        if !resp.status().is_success() {
            tracing::warn!("warmup #{i}: status {}", resp.status());
            let _ = resp.into_body().collect().await;
            break;
        }
        if let Err(e) = resp.into_body().collect().await {
            tracing::warn!("warmup #{i}: drain failed: {e:#}");
            break;
        }
        let dt = start.elapsed().as_millis() as u64;
        elapsed_ms.push(dt);
        completed += 1;
        tracing::info!("warmup #{i}: {dt} ms");
    }

    WarmupResult {
        requested: count,
        completed,
        elapsed_ms,
    }
}

async fn discover_model(child_port: u16) -> Result<String> {
    let client: Client<HttpConnector, Empty<Bytes>> =
        Client::builder(TokioExecutor::new()).build_http();
    let req = Request::builder()
        .method("GET")
        .uri(format!("http://127.0.0.1:{child_port}/v1/models"))
        .body(Empty::<Bytes>::new())?;
    let resp = client.request(req).await?;
    let body = resp.into_body().collect().await?.to_bytes();
    let v: Value = serde_json::from_slice(&body)?;
    let id = v
        .pointer("/data/0/id")
        .and_then(|x| x.as_str())
        .ok_or_else(|| anyhow!("/v1/models missing data[0].id: {v}"))?;
    Ok(id.to_string())
}

// ---------------------------------------------------------------------------
// Top-level assembly
// ---------------------------------------------------------------------------

pub async fn engine_version(engine: Engine, venv: Option<PathBuf>) -> Option<String> {
    let bin = match (engine, &venv) {
        (Engine::Sglang, Some(v)) => v.join("bin/python"),
        (Engine::Vllm, Some(v)) => v.join("bin/vllm"),
        (Engine::TrtLlm, Some(v)) => v.join("bin/python"),
        _ => return None,
    };
    match engine {
        Engine::Sglang => run_capture(
            bin.to_string_lossy().as_ref(),
            &["-c", "import sglang; print(sglang.__version__)"],
        )
        .await
        .ok()
        .map(|s| s.trim().to_string()),
        Engine::Vllm => run_capture(bin.to_string_lossy().as_ref(), &["--version"])
            .await
            .ok()
            .map(|s| s.trim().to_string()),
        Engine::TrtLlm => run_capture(
            bin.to_string_lossy().as_ref(),
            &[
                "-c",
                "import tensorrt_llm; print(tensorrt_llm.__version__)",
            ],
        )
        .await
        .ok()
        .map(|s| s.trim().to_string()),
    }
}

pub fn config_section(cfg: &ResolvedConfig) -> ConfigSection {
    ConfigSection {
        path: cfg.path.clone(),
        sha256: cfg.raw_sha256.clone(),
        raw: cfg.raw.clone(),
        description: cfg.description.clone(),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn run_capture(cmd: &str, args: &[&str]) -> Result<String> {
    let out = tokio::time::timeout(Duration::from_secs(60), Command::new(cmd).args(args).output())
        .await
        .with_context(|| format!("`{cmd}` timed out after 60s"))??;
    if !out.status.success() {
        return Err(anyhow!(
            "`{cmd}` exited {:?}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}
