use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use indexmap::IndexMap;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Engine {
    Sglang,
    Vllm,
    #[serde(alias = "trt-llm", alias = "trtllm")]
    TrtLlm,
}

impl Engine {
    pub fn as_str(self) -> &'static str {
        match self {
            Engine::Sglang => "sglang",
            Engine::Vllm => "vllm",
            Engine::TrtLlm => "trt-llm",
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum EnvValue {
    Plain(String),
    Secret { secret: String },
}

#[derive(Debug, Deserialize)]
pub struct RawConfig {
    pub description: String,
    pub engine: Engine,
    pub cmd: String,
    pub args: Vec<String>,
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    #[serde(default)]
    pub venv: Option<PathBuf>,
    #[serde(default = "default_warmup")]
    pub warmup: u32,
    pub env: IndexMap<String, EnvValue>,
}

fn default_warmup() -> u32 {
    3
}

pub struct ResolvedConfig {
    pub path: PathBuf,
    pub raw: String,
    pub raw_sha256: String,
    pub description: String,
    pub engine: Engine,
    pub cmd: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub venv: Option<PathBuf>,
    pub warmup: u32,
    /// Final, resolved env to pass to the child. Insertion order preserved
    /// so /meta reflects YAML order for humans.
    pub child_env: IndexMap<String, String>,
    /// Keys whose value the YAML marked as `{ secret: ... }`. Used to
    /// redact those keys when reading /proc/<pid>/environ for /meta.
    pub secret_keys: HashSet<String>,
}

pub fn load(path: &Path) -> Result<ResolvedConfig> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading config: {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(raw.as_bytes());
    let raw_sha256 = hex::encode(hasher.finalize());

    let cfg: RawConfig = serde_yaml_ng::from_str(&raw)
        .with_context(|| format!("parsing YAML: {}", path.display()))?;

    if cfg.description.trim().is_empty() {
        bail!("`description` must be non-empty");
    }
    for a in &cfg.args {
        if a == "--host"
            || a == "--port"
            || a.starts_with("--host=")
            || a.starts_with("--port=")
        {
            bail!(
                "`args` must not contain --host or --port; the proxy injects \
                 those (saw {a:?})"
            );
        }
    }

    let mut child_env = IndexMap::with_capacity(cfg.env.len());
    let mut secret_keys = HashSet::new();
    for (k, v) in &cfg.env {
        let raw_value = match v {
            EnvValue::Plain(s) => s.clone(),
            EnvValue::Secret { secret } => {
                secret_keys.insert(k.clone());
                secret.clone()
            }
        };
        let resolved = interpolate(&raw_value)
            .with_context(|| format!("env.{k}"))?;
        child_env.insert(k.clone(), resolved);
    }

    Ok(ResolvedConfig {
        path: path.to_path_buf(),
        raw,
        raw_sha256,
        description: cfg.description,
        engine: cfg.engine,
        cmd: cfg.cmd,
        args: cfg.args,
        cwd: cfg.cwd,
        venv: cfg.venv,
        warmup: cfg.warmup,
        child_env,
        secret_keys,
    })
}

/// Replace `$VAR` and `${VAR}` from the proxy's runtime env. Hard-error
/// if a referenced var is unset — silent fallthrough is the kind of
/// invisible-config-drift this whole project exists to prevent.
fn interpolate(s: &str) -> Result<String> {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    let mut copied_to = 0;
    while i < bytes.len() {
        if bytes[i] != b'$' {
            i += 1;
            continue;
        }
        out.push_str(&s[copied_to..i]);
        if i + 1 >= bytes.len() {
            out.push('$');
            i += 1;
            copied_to = i;
            continue;
        }
        match bytes[i + 1] {
            b'{' => {
                let rest = &s[i + 2..];
                let end = rest
                    .find('}')
                    .ok_or_else(|| anyhow!("unterminated ${{...}} in {s:?}"))?;
                let name = &rest[..end];
                let val = std::env::var(name).with_context(|| {
                    format!("env var ${name} referenced in YAML but not set in proxy env")
                })?;
                out.push_str(&val);
                i = i + 2 + end + 1;
                copied_to = i;
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                let mut j = i + 1;
                while j < bytes.len() {
                    let b = bytes[j];
                    if b.is_ascii_alphanumeric() || b == b'_' {
                        j += 1;
                    } else {
                        break;
                    }
                }
                let name = &s[i + 1..j];
                let val = std::env::var(name).with_context(|| {
                    format!("env var ${name} referenced in YAML but not set in proxy env")
                })?;
                out.push_str(&val);
                i = j;
                copied_to = i;
            }
            _ => {
                out.push('$');
                i += 1;
                copied_to = i;
            }
        }
    }
    out.push_str(&s[copied_to..]);
    Ok(out)
}
