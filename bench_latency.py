#!/usr/bin/env python3
"""
Benchmark an OpenAI-compatible chat completions endpoint using queries sampled
from queries.csv.

Usage:
  python bench_latency.py config.json

Each run creates a UUID-named subfolder inside cfg.output_dir containing:
  - result.json  JSONL event log; each line is one of RequestEvent, ChunkEvent,
                 FinalEvent, or ErrorEvent from schema.py. `t_s` on every event
                 is relative to start-of-program (a single global t0).
  - meta.json    Sidecar with `started_unix` and the full config (RunMeta in
                 schema.py). Server hardware / run tags belong in cfg.metadata.

Before the measured runs, one warmup request is sent to load the model. The
warmup is consumed fully but not logged.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, TextIO

import httpx

from schema import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChunkEvent,
    Config,
    ErrorEvent,
    FinalChatCompletionChunk,
    FinalEvent,
    LogEvent,
    RequestEvent,
    RunMeta,
)


def load_config(path: str) -> Config:
    with open(path) as f:
        return Config.model_validate(json.load(f))


def load_queries(
    path: str,
    n: int,
    seed: int,
    model_override: str,
    extra_body: dict[str, Any],
) -> list[dict[str, Any]]:
    # queries.csv has a single 'data' column whose value is a JSON string
    # representing an OpenAI-style chat completion request.
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r["data"] for r in reader]

    rng = random.Random(seed)
    sampled = rng.sample(rows, min(n, len(rows)))

    auto_guided_json = bool(extra_body.pop("_auto_guided_json", False))

    out: list[dict[str, Any]] = []
    for raw in sampled:
        payload = json.loads(raw)
        payload["model"] = model_override
        # queries.csv carries a non-OpenAI `provider` field (e.g. "groq")
        # that the original generator embedded. vLLM/sglang silently accept
        # unknown fields; trt-llm uses strict pydantic and 400s the request.
        # Strip it here so all engines see the same OpenAI-spec body.
        payload.pop("provider", None)
        # Per-query JSON guided-decoding heuristic: 84% of queries.csv
        # system prompts say "Return ONLY valid JSON" / similar. Setting
        # guided_json={"type":"object"} on those lets vllm's grammar
        # backend (xgrammar by default in 0.20) skip LM forward passes
        # on tokens that are uniquely determined by the JSON grammar
        # (~47% of output is pure JSON syntax for this workload).
        # Triggered only when extra_body has _auto_guided_json: true so
        # baseline runs stay bit-identical.
        if auto_guided_json:
            sys_msg = next(
                (m.get("content", "") for m in payload.get("messages", [])
                 if m.get("role") == "system"),
                "",
            )
            if "JSON" in sys_msg or "json" in sys_msg.split("\n", 1)[0].lower():
                payload.setdefault("extra_body", {})
                payload["extra_body"]["guided_json"] = {"type": "object"}
        # Merge extra_body on top — caller-provided overrides win.
        payload.update(extra_body)
        out.append(payload)
    return out


LogFn = Callable[[LogEvent], Awaitable[None]]
ClockFn = Callable[[], float]


def make_clock(t0: float) -> ClockFn:
    def now() -> float:
        return time.perf_counter() - t0
    return now


def make_logger(file: TextIO, lock: asyncio.Lock) -> LogFn:
    async def log(event: LogEvent) -> None:
        line = event.model_dump_json()
        async with lock:
            file.write(line + "\n")
    return log


async def warmup(
    url: str,
    headers: dict[str, str],
    model: str,
    extra_body: dict[str, Any],
    timeout: float,
    verbose: bool,
) -> None:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Tell me a short story."}],
        "stream": True,
    }
    body.update(extra_body)
    timeout_cfg = httpx.Timeout(timeout, connect=timeout)
    if verbose:
        print("warming up...", file=sys.stderr)
    try:
        async with httpx.AsyncClient(timeout=timeout_cfg, http2=False) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", "replace")[:500]
                    print(
                        f"warmup failed: HTTP {resp.status_code}: {text}",
                        file=sys.stderr,
                    )
                    return
                async for _ in resp.aiter_lines():
                    pass
        if verbose:
            print("warmup done", file=sys.stderr)
    except Exception as e:
        print(f"warmup failed: {type(e).__name__}: {e}", file=sys.stderr)


async def run_one(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    concurrency: int,
    request_id: int,
    now: ClockFn,
    log: LogFn,
) -> str | None:
    body = {**payload, "stream": True}
    # Ask for usage in the final chunk when the server supports it (OpenAI,
    # vLLM, llama.cpp server). Harmless if ignored.
    body.setdefault("stream_options", {"include_usage": True})

    await log(RequestEvent(
        t_s=now(),
        request_id=request_id,
        concurrency=concurrency,
        body=ChatCompletionRequest.model_validate(body),
    ))

    # The last chunk gets emitted as FinalEvent rather than ChunkEvent, so
    # buffer-and-shift instead of emitting on arrival.
    buf_chunk: ChatCompletionChunk | None = None
    buf_t: float = 0.0
    status: int | None = None
    error: str | None = None

    try:
        async with client.stream("POST", url, headers=headers, json=body) as resp:
            status = resp.status_code
            if resp.status_code >= 400:
                text = (await resp.aread()).decode("utf-8", "replace")[:500]
                error = f"HTTP {status}: {text}"
            else:
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].lstrip()
                    if data_str == "[DONE]":
                        continue
                    try:
                        raw = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    chunk_t = now()
                    chunk = ChatCompletionChunk.model_validate(raw)
                    if buf_chunk is not None:
                        await log(ChunkEvent(
                            t_s=buf_t,
                            request_id=request_id,
                            chunk=buf_chunk,
                        ))
                    buf_chunk = chunk
                    buf_t = chunk_t
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    if error is None and buf_chunk is not None and buf_chunk.usage is not None:
        assert status is not None  # set on the success branch above
        await log(FinalEvent(
            t_s=buf_t,
            request_id=request_id,
            chunk=FinalChatCompletionChunk.model_validate(buf_chunk.model_dump()),
            status=status,
        ))
        return None

    if buf_chunk is not None:
        await log(ChunkEvent(
            t_s=buf_t, request_id=request_id, chunk=buf_chunk,
        ))
    if error is None:
        # Stream closed without an error and without a final chunk that has
        # usage — most likely the server didn't honor stream_options.include_usage.
        error = "stream ended without a final chunk carrying usage"
    await log(ErrorEvent(
        t_s=now(),
        request_id=request_id,
        status=status,
        error=error,
    ))
    return error


async def run_at_concurrency(
    url: str,
    headers: dict[str, str],
    payloads: list[dict[str, Any]],
    concurrency: int,
    timeout: float,
    base_id: int,
    now: ClockFn,
    log: LogFn,
) -> list[str | None]:
    # Size the pool to the concurrency so we don't queue inside httpx.
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    timeout_cfg = httpx.Timeout(timeout, connect=timeout)
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(limits=limits, timeout=timeout_cfg, http2=False) as client:
        async def bounded(i: int, p: dict[str, Any]) -> str | None:
            async with sem:
                return await run_one(
                    client, url, headers, p, concurrency, base_id + i, now, log,
                )

        return await asyncio.gather(
            *(bounded(i, p) for i, p in enumerate(payloads))
        )


async def main_async(cfg: Config, message: str, t0: float, verbose: bool) -> None:
    queries = load_queries(
        cfg.queries,
        cfg.num_queries,
        cfg.seed,
        cfg.model,
        cfg.extra_body,
    )
    print(
        f"loaded {len(queries)} queries from {cfg.queries} "
        f"(seed={cfg.seed}, model={queries[0].get('model')})",
        file=sys.stderr,
    )

    run_dir = Path(cfg.output_dir) / str(uuid.uuid4())
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"run dir: {run_dir}", file=sys.stderr)

    meta = RunMeta(started_unix=time.time(), message=message, config=cfg)
    (run_dir / "meta.json").write_text(meta.model_dump_json(indent=2))

    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }

    await warmup(url, headers, cfg.model, cfg.extra_body, cfg.timeout, verbose)

    now = make_clock(t0)

    with open(run_dir / "result.json", "w") as f:
        lock = asyncio.Lock()
        log = make_logger(f, lock)

        next_id = 0
        for c in cfg.concurrency:
            print(f"\n--- running concurrency={c} ---", file=sys.stderr)
            errors = await run_at_concurrency(
                url, headers, queries, c, cfg.timeout, next_id, now, log,
            )
            next_id += len(queries)

            errs = [e for e in errors if e]
            print(
                f"  {len(queries) - len(errs)}/{len(queries)} ok",
                file=sys.stderr,
            )
            # Surface the first few errors so misconfig is obvious.
            for e in errs[:3]:
                print(f"  error: {e}", file=sys.stderr)


def main() -> None:
    t0 = time.perf_counter()
    p = argparse.ArgumentParser(
        usage='python bench_latency.py config.json "<message>" [-v]'
    )
    p.add_argument("config", help="path to config.json")
    p.add_argument(
        "message",
        help="free-form description of the run, e.g. 'sglang with speculative decoding'",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="print warmup status to stderr",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    try:
        asyncio.run(main_async(cfg, args.message, t0, args.verbose))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
