#!/usr/bin/env python3
"""
Benchmark an OpenAI-compatible chat completions endpoint using queries sampled
from queries.csv.

Measures:
  - TTFT (time to first streamed token)
  - Total request latency
  - Output tokens (when the server reports usage)

Example:
  python bench_latency.py \\
      --base-url http://localhost:8000/v1 \\
      -n 64 \\
      --concurrency 1 2 4 8 16 32 \\
      --model openai/gpt-oss-120b \\
      --extra-body '{"reasoning_effort": "low"}' \\
      --output report.json

Requires: httpx  (pip install httpx)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Loading queries
# ---------------------------------------------------------------------------

def load_queries(
    path: str,
    n: int,
    seed: int,
    model_override: str | None,
    extra_body: dict[str, Any],
    drop_keys: list[str],
) -> list[dict[str, Any]]:
    # queries.csv has a single 'data' column whose value is a JSON string
    # representing an OpenAI-style chat completion request.
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r["data"] for r in reader]

    rng = random.Random(seed)
    sampled = rng.sample(rows, min(n, len(rows)))

    out: list[dict[str, Any]] = []
    for raw in sampled:
        payload = json.loads(raw)
        for k in drop_keys:
            payload.pop(k, None)
        if model_override:
            payload["model"] = model_override
        # Merge extra_body on top — caller-provided overrides win.
        payload.update(extra_body)
        out.append(payload)
    return out


# ---------------------------------------------------------------------------
# Single-request timing
# ---------------------------------------------------------------------------

@dataclass
class Result:
    concurrency: int
    request_idx: int
    ttft_s: float | None
    total_latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    status: int | None
    error: str | None


def _delta_has_token(chunk: dict[str, Any]) -> bool:
    # Consider first "token" any delta with meaningful textual output or a
    # reasoning / tool-call fragment. Empty role-only chunks don't count.
    choices = chunk.get("choices") or []
    if not choices:
        return False
    delta = choices[0].get("delta") or {}
    for k in ("content", "reasoning", "reasoning_content"):
        v = delta.get(k)
        if isinstance(v, str) and v:
            return True
    if delta.get("tool_calls"):
        return True
    return False


async def run_one(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    concurrency: int,
    idx: int,
) -> Result:
    body = {**payload, "stream": True}
    # Ask for usage in the final chunk when the server supports it (OpenAI,
    # vLLM, llama.cpp server). Harmless if ignored.
    body.setdefault("stream_options", {"include_usage": True})

    ttft: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    status: int | None = None

    start = time.perf_counter()
    try:
        async with client.stream("POST", url, headers=headers, json=body) as resp:
            status = resp.status_code
            if resp.status_code >= 400:
                text = (await resp.aread()).decode("utf-8", "replace")[:500]
                return Result(
                    concurrency, idx, None, time.perf_counter() - start,
                    None, None, status, f"HTTP {status}: {text}",
                )
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].lstrip()
                if data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if ttft is None and _delta_has_token(chunk):
                    ttft = time.perf_counter() - start
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get(
                        "completion_tokens", completion_tokens
                    )
        total = time.perf_counter() - start
        return Result(
            concurrency, idx, ttft, total, prompt_tokens, completion_tokens,
            status, None,
        )
    except Exception as e:
        return Result(
            concurrency, idx, ttft, time.perf_counter() - start,
            prompt_tokens, completion_tokens, status, f"{type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# Concurrency runner
# ---------------------------------------------------------------------------

async def run_at_concurrency(
    url: str,
    headers: dict[str, str],
    payloads: list[dict[str, Any]],
    concurrency: int,
    timeout: float,
) -> tuple[list[Result], float]:
    # Size the pool to the concurrency so we don't queue inside httpx.
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    timeout_cfg = httpx.Timeout(timeout, connect=timeout)
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(limits=limits, timeout=timeout_cfg, http2=False) as client:
        async def bounded(i: int, p: dict[str, Any]) -> Result:
            async with sem:
                return await run_one(client, url, headers, p, concurrency, i)

        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(bounded(i, p) for i, p in enumerate(payloads))
        )
        wallclock = time.perf_counter() - t0
    return results, wallclock


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------

def _pcts(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {}
    vs = sorted(vals)
    def q(p: float) -> float:
        if len(vs) == 1:
            return vs[0]
        k = (len(vs) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(vs) - 1)
        return vs[lo] + (vs[hi] - vs[lo]) * (k - lo)
    return {
        "n": len(vs),
        "mean": statistics.fmean(vs),
        "min": vs[0],
        "p50": q(0.5),
        "p90": q(0.9),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": vs[-1],
    }


@dataclass
class Summary:
    concurrency: int
    n: int
    errors: int
    wallclock_s: float
    throughput_rps: float
    ttft_s: dict[str, float] = field(default_factory=dict)
    total_latency_s: dict[str, float] = field(default_factory=dict)
    completion_tokens: dict[str, float] = field(default_factory=dict)
    tokens_per_second: dict[str, float] = field(default_factory=dict)


def summarize(results: list[Result], concurrency: int, wallclock: float) -> Summary:
    ok = [r for r in results if r.error is None]
    errs = [r for r in results if r.error is not None]
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    totals = [r.total_latency_s for r in ok]
    tokens = [r.completion_tokens for r in ok if r.completion_tokens]
    tps = [
        r.completion_tokens / r.total_latency_s
        for r in ok
        if r.completion_tokens and r.total_latency_s > 0
    ]
    return Summary(
        concurrency=concurrency,
        n=len(results),
        errors=len(errs),
        wallclock_s=wallclock,
        throughput_rps=(len(ok) / wallclock) if wallclock > 0 else 0.0,
        ttft_s=_pcts(ttfts),
        total_latency_s=_pcts(totals),
        completion_tokens=_pcts([float(t) for t in tokens]),
        tokens_per_second=_pcts(tps),
    )


def print_summary(s: Summary) -> None:
    line = (
        f"concurrency={s.concurrency:>3}  n={s.n:>3}  errors={s.errors:>2}  "
        f"wall={s.wallclock_s:6.2f}s  throughput={s.throughput_rps:5.2f} rps"
    )
    print(line)
    for label, d in (
        ("ttft   ", s.ttft_s),
        ("total  ", s.total_latency_s),
        ("tok/s  ", s.tokens_per_second),
        ("out_tok", s.completion_tokens),
    ):
        if not d:
            continue
        parts = [
            f"mean={d['mean']:.3f}",
            f"p50={d['p50']:.3f}",
            f"p90={d['p90']:.3f}",
            f"p99={d['p99']:.3f}",
            f"max={d['max']:.3f}",
        ]
        print(f"  {label}: " + "  ".join(parts))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    extra_body = json.loads(args.extra_body) if args.extra_body else {}
    queries = load_queries(
        args.queries,
        args.num_queries,
        args.seed,
        args.model,
        extra_body,
        drop_keys=args.drop_key,
    )
    print(
        f"loaded {len(queries)} queries from {args.queries} "
        f"(seed={args.seed}, model={queries[0].get('model')})",
        file=sys.stderr,
    )

    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }

    summaries: list[Summary] = []
    all_requests: list[dict[str, Any]] = []

    for c in args.concurrency:
        print(f"\n--- running concurrency={c} ---", file=sys.stderr)
        # Reuse the same sampled queries at every level for comparability.
        # If the server caches on exact prompts, pass --reshuffle to re-sample
        # per level.
        payloads = queries
        if args.reshuffle:
            rng = random.Random(args.seed + c)
            payloads = rng.sample(queries, len(queries))

        results, wallclock = await run_at_concurrency(
            url, headers, payloads, c, args.timeout
        )
        s = summarize(results, c, wallclock)
        summaries.append(s)
        all_requests.extend(asdict(r) for r in results)
        print_summary(s)

        # Surface the first few errors so misconfig is obvious.
        errs = [r for r in results if r.error]
        for r in errs[:3]:
            print(f"  error idx={r.request_idx}: {r.error}", file=sys.stderr)

    if args.output:
        report = {
            "config": {k: v for k, v in vars(args).items()},
            "summaries": [asdict(s) for s in summaries],
            "requests": all_requests,
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.output}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queries", default="queries.csv", help="path to queries.csv")
    p.add_argument("--base-url", default="http://localhost:8000/v1",
                   help="OpenAI-compatible base URL (must include /v1 if the server expects it)")
    p.add_argument("--api-key", default="sk-noop")
    p.add_argument("-n", "--num-queries", type=int, default=64,
                   help="how many queries to sample from the CSV")
    p.add_argument("--concurrency", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--model", default=None,
                   help="override the 'model' field on every sampled request")
    p.add_argument("--extra-body", default=None,
                   help="JSON object merged into every request body "
                        "(e.g. '{\"reasoning_effort\":\"low\",\"temperature\":0}')")
    p.add_argument("--drop-key", action="append", default=["provider"],
                   help="keys to strip from each payload (repeatable). "
                        "Default drops 'provider' since local servers don't want it.")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reshuffle", action="store_true",
                   help="re-sample the order per concurrency level (helps if "
                        "the server caches exact prompts across runs)")
    p.add_argument("--output", default=None,
                   help="write full JSON report (summaries + per-request)")
    return p.parse_args()


def main() -> None:
    try:
        asyncio.run(main_async(parse_args()))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
