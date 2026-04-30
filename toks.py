#!/usr/bin/env python3
"""Print median tokens/second from a bench_latency.py JSONL output.

Usage: python toks.py report.jsonl

For each successful request, throughput = chunk.usage.completion_tokens
                                          / (FinalEvent.t_s - RequestEvent.t_s).
Errored requests are skipped.
"""
from __future__ import annotations

import statistics
import sys

from schema import FinalEvent, RequestEvent, iter_events


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python toks.py report.jsonl", file=sys.stderr)
        sys.exit(2)

    requests: dict[int, float] = {}
    finals: dict[int, FinalEvent] = {}

    for ev in iter_events(sys.argv[1]):
        if isinstance(ev, RequestEvent):
            requests[ev.request_id] = ev.t_s
        elif isinstance(ev, FinalEvent):
            finals[ev.request_id] = ev

    vals: list[float] = []
    for rid, final in finals.items():
        if rid not in requests:
            continue
        ct = final.chunk.usage.completion_tokens
        if not ct:
            continue
        dur = final.t_s - requests[rid]
        if dur > 0:
            vals.append(ct / dur)

    print(statistics.median(vals))


if __name__ == "__main__":
    main()
