#!/usr/bin/env bash
# Run one bench iteration: spawn bench-proxy with the given YAML, wait for /meta,
# run bench_latency.py, parse median tok/s, kill proxy. Print one number to stdout.
#
# Usage:
#   bin/verify_bench.sh <yaml-path>
#   bin/verify_bench.sh                 # uses configs/.current_experiment
#
# Env knobs:
#   NUM_QUERIES (default 50), CONCURRENCY (default 1),
#   READY_TIMEOUT_S (default 1500), BENCH_TIMEOUT_S (default 600),
#   MODEL (default openai/gpt-oss-120b — must match the model the engine serves).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

YAML="${1:-}"
if [ -z "$YAML" ]; then
  PTR="$REPO_ROOT/configs/.current_experiment"
  [ -f "$PTR" ] || { echo "no YAML arg and no configs/.current_experiment pointer" >&2; exit 1; }
  YAML="$REPO_ROOT/$(cat "$PTR")"
fi
[ -f "$YAML" ] || { echo "missing YAML: $YAML" >&2; exit 1; }

PROXY_BIN="$REPO_ROOT/proxy/target/release/bench-proxy"
[ -x "$PROXY_BIN" ] || { echo "bench-proxy not built: $PROXY_BIN — run cargo build --release inside a devshell" >&2; exit 1; }

ENGINE=$(awk '/^engine:/ {print $2; exit}' "$YAML" | tr -d '"' )
case "$ENGINE" in
  sglang|vllm|trt-llm) ;;
  *) echo "unknown engine '$ENGINE' in $YAML" >&2; exit 1 ;;
esac

VENV="$REPO_ROOT/$ENGINE/.venv"
[ -x "$VENV/bin/python" ] || { echo "engine venv missing: $VENV — run 'uv sync' first" >&2; exit 1; }

NUM_QUERIES="${NUM_QUERIES:-50}"
CONCURRENCY="${CONCURRENCY:-1}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1500}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-600}"
MODEL="${MODEL:-openai/gpt-oss-120b}"

# Pick a free port for the proxy listener.
LISTEN_PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()' 2>/dev/null \
  || $VENV/bin/python -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')

STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$(basename "$YAML" .yaml)"
RUN_ROOT="$REPO_ROOT/.bench_runs/$STAMP"
mkdir -p "$RUN_ROOT"
PROXY_LOG="$RUN_ROOT/proxy.log"
BENCH_LOG="$RUN_ROOT/bench.log"
BENCH_CFG="$RUN_ROOT/bench.json"

cat > "$BENCH_CFG" <<EOF
{
  "queries": "$REPO_ROOT/queries.csv",
  "base_url": "http://127.0.0.1:$LISTEN_PORT/v1",
  "api_key": "sk-empty",
  "num_queries": $NUM_QUERIES,
  "concurrency": [$CONCURRENCY],
  "model": "$MODEL",
  "extra_body": {},
  "timeout": $BENCH_TIMEOUT_S,
  "seed": 42,
  "output_dir": "$RUN_ROOT",
  "metadata": {"yaml": "$(basename "$YAML")", "host": "b200-singleton"}
}
EOF

PROXY_PID=""
cleanup() {
  if [ -n "$PROXY_PID" ] && kill -0 "$PROXY_PID" 2>/dev/null; then
    kill -TERM "$PROXY_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$PROXY_PID" 2>/dev/null || break
      sleep 0.5
    done
    kill -KILL "$PROXY_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Spawn proxy inside the engine devshell so $NIX_LD_LIBRARY_PATH and friends
# are populated for the YAML's env interpolation.
echo "[verify] yaml=$YAML engine=$ENGINE listen=127.0.0.1:$LISTEN_PORT run=$RUN_ROOT" >&2
nix develop "$REPO_ROOT#$ENGINE" --command \
  "$PROXY_BIN" \
    --listen "127.0.0.1:$LISTEN_PORT" \
    --config "$YAML" \
    --repo-root "$REPO_ROOT" \
    --ready-timeout-secs "$READY_TIMEOUT_S" \
  > "$PROXY_LOG" 2>&1 &
PROXY_PID=$!

# Wait for proxy /meta to return 200 (means engine ready + warmup done + snapshot published).
DEADLINE=$(( $(date +%s) + READY_TIMEOUT_S + 60 ))
while :; do
  if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "[verify] proxy died during startup — tail of proxy.log:" >&2
    tail -80 "$PROXY_LOG" >&2 || true
    exit 2
  fi
  if curl -fsS -m 3 "http://127.0.0.1:$LISTEN_PORT/meta" -o "$RUN_ROOT/meta.json" 2>/dev/null; then
    echo "[verify] /meta is up — engine ready" >&2
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "[verify] /meta did not become ready before deadline; tail of proxy.log:" >&2
    tail -80 "$PROXY_LOG" >&2 || true
    exit 3
  fi
  sleep 2
done

# Run bench from inside the engine venv (it has httpx + pydantic).
"$VENV/bin/python" "$REPO_ROOT/bench_latency.py" "$BENCH_CFG" "$(basename "$YAML")" \
  > "$BENCH_LOG" 2>&1

# bench_latency.py creates a UUID subdir under output_dir; find result.json.
RESULT_JSON=$(find "$RUN_ROOT" -mindepth 2 -maxdepth 2 -name result.json 2>/dev/null | head -1 || true)
if [ -z "$RESULT_JSON" ] || [ ! -f "$RESULT_JSON" ]; then
  echo "[verify] no result.json in $RUN_ROOT — tail of bench.log:" >&2
  tail -80 "$BENCH_LOG" >&2 || true
  exit 4
fi

# Print median tok/s (the metric).
"$VENV/bin/python" "$REPO_ROOT/toks.py" "$RESULT_JSON"
