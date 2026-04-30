#!/usr/bin/env python3
"""Print median TTFT (seconds) from a bench_latency.py JSONL output.

Usage: python ttft.py report.jsonl

TTFT = (t_s of the first response carrying a token-bearing delta) - (RequestEvent.t_s).
Errored requests are skipped.
"""
from __future__ import annotations

import statistics
import sys

from schema import (
    ChatCompletionChunk,
    ChunkEvent,
    ErrorEvent,
    FinalEvent,
    RequestEvent,
    iter_events,
)


def has_token(chunk: ChatCompletionChunk) -> bool:
    for choice in chunk.choices:
        delta = choice.delta
        if delta.content or delta.tool_calls or delta.refusal:
            return True
        # Reasoning-capable servers emit the first token in extras like
        # "reasoning" / "reasoning_content".
        extras = delta.model_extra or {}
        for k in ("reasoning", "reasoning_content"):
            v = extras.get(k)
            if isinstance(v, str) and v:
                return True
    return False


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python ttft.py report.jsonl", file=sys.stderr)
        sys.exit(2)

    requests: dict[int, float] = {}
    firsts: dict[int, float] = {}
    errored: set[int] = set()

    for ev in iter_events(sys.argv[1]):
        if isinstance(ev, RequestEvent):
            requests[ev.request_id] = ev.t_s
        elif isinstance(ev, ChunkEvent):
            if ev.request_id in firsts:
                continue
            if has_token(ev.chunk):
                firsts[ev.request_id] = ev.t_s
        elif isinstance(ev, FinalEvent):
            if ev.request_id not in firsts and has_token(ev.chunk):
                firsts[ev.request_id] = ev.t_s
        elif isinstance(ev, ErrorEvent):
            errored.add(ev.request_id)

    vals = [
        firsts[rid] - requests[rid]
        for rid in firsts
        if rid in requests and rid not in errored
    ]
    print(statistics.median(vals))


if __name__ == "__main__":
    main()
