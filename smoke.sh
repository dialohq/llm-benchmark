#!/usr/bin/env bash
# Hello-world smoke test: launch the engine's OpenAI-compatible server,
# wait until it accepts requests, fire a single chat completion via curl,
# then shut down.
#
# Run inside the matching `nix develop .#<engine>` shell. The shell sets
# CUDA_HOME / LD_LIBRARY_PATH / HF_HUB_ENABLE_HF_TRANSFER for us.
#
#   nix develop .#sglang  --command bash smoke.sh sglang
#   nix develop .#vllm    --command bash smoke.sh vllm
#   nix develop .#trt-llm --command bash smoke.sh trt-llm
#
# Optimisations are intentionally OFF (eager mode, no CUDA graphs,
# moderate KV-cache budget) — we only want to prove the engine boots
# and serves a token.

set -euo pipefail

ENGINE="${1:-}"
MODEL="${SMOKE_MODEL:-openai/gpt-oss-120b}"
PROMPT="${SMOKE_PROMPT:-Hello, my name is}"
MAX_TOKENS="${SMOKE_MAX_TOKENS:-32}"
TP="${SMOKE_TP:-2}"
HOST=127.0.0.1
LOG_DIR="${LOG_DIR:-/tmp/llm-bench-smoke}"
mkdir -p "$LOG_DIR"

case "$ENGINE" in
  sglang)
    PORT=30000
    LAUNCH=(
      python -m sglang.launch_server
      --model-path "$MODEL"
      --tp "$TP"
      --mem-fraction-static 0.80
      --disable-cuda-graph
      --host "$HOST" --port "$PORT"
    )
    ;;
  vllm)
    PORT=8000
    LAUNCH=(
      vllm serve "$MODEL"
      --tensor-parallel-size "$TP"
      --enforce-eager
      --gpu-memory-utilization 0.80
      --max-model-len 2048
      --host "$HOST" --port "$PORT"
    )
    ;;
  trt-llm)
    PORT=8000
    LAUNCH=(
      trtllm-serve "$MODEL"
      --tp_size "$TP"
      --host "$HOST" --port "$PORT"
    )
    ;;
  *)
    echo "usage: $0 {sglang|vllm|trt-llm}" >&2
    exit 2
    ;;
esac

LOG="$LOG_DIR/$ENGINE.log"
: > "$LOG"
echo "[$ENGINE] launch: ${LAUNCH[*]}"
echo "[$ENGINE] log:    $LOG"
"${LAUNCH[@]}" >>"$LOG" 2>&1 &
SERVER_PID=$!
trap 'echo "[$ENGINE] shutdown pid=$SERVER_PID"; kill -INT $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT

# Readiness probe — every engine exposes /v1/models on success and /health
# during boot. We poll /v1/models because it returns 503 until the weights
# are loaded, which is what we actually care about.
URL_BASE="http://$HOST:$PORT"
DEADLINE=$(( $(date +%s) + 1200 ))   # 20 min — gpt-oss-120b weight load is slow
echo "[$ENGINE] waiting for $URL_BASE/v1/models (timeout 20m) ..."
while true; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[$ENGINE] server died before ready — last log lines:" >&2
    tail -30 "$LOG" >&2
    exit 1
  fi
  if curl -fsS "$URL_BASE/v1/models" >/dev/null 2>&1; then
    echo "[$ENGINE] ready"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "[$ENGINE] timeout waiting for ready" >&2
    tail -30 "$LOG" >&2
    exit 1
  fi
  sleep 5
done

echo "[$ENGINE] POST /v1/chat/completions"
curl -sS "$URL_BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(cat <<JSON
{
  "model": "$MODEL",
  "messages": [{"role": "user", "content": "$PROMPT"}],
  "temperature": 0.0,
  "max_tokens": $MAX_TOKENS
}
JSON
)" | tee "$LOG_DIR/$ENGINE.response.json"
echo

echo "[$ENGINE] done — full server log: $LOG"
