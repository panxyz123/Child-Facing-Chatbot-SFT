"""Summarize agent traces (logs/traces.jsonl): traffic mix, guard and retrieval decisions, latency per node.

    python scripts/summarize_traces.py                         # all traces
    python scripts/summarize_traces.py --since 2026-11-20T00:00
"""
import argparse
import json

import numpy as np
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="logs/traces.jsonl")
    ap.add_argument("--since", default=None, help="ISO time; only traces at or after it")
    a = ap.parse_args(argv)
    T = pd.DataFrame([json.loads(l) for l in open(a.path, encoding="utf-8")])
    if a.since:
        T = T[T.ts >= a.since]
    print(f"{len(T)} requests, {T.session.nunique()} sessions")

    print("\nroute:", T.route.value_counts(normalize=True).round(3).to_dict())
    print("guard:", T.guard.fillna("none").value_counts().to_dict())
    k = T[T.route == "knowledge"]
    if len(k):
        print("gate (knowledge questions):", k.gate.value_counts().to_dict())
    print("output flags:", T.output_flag.fillna("none").value_counts().to_dict())
    print(f"regenerated: {(T.regenerations > 0).mean():.1%}")

    nodes = pd.DataFrame(T.node_ms.tolist())
    rows = {c: dict(n=int(nodes[c].notna().sum()), p50=np.nanpercentile(nodes[c], 50), p95=np.nanpercentile(nodes[c], 95),
                    mean=nodes[c].mean()) for c in nodes.columns}
    rows["total"] = dict(n=len(T), p50=T.latency_ms.quantile(.5), p95=T.latency_ms.quantile(.95), mean=T.latency_ms.mean())
    print("\nlatency (ms)\n" + pd.DataFrame(rows).T.round(1).to_string())

    answered = T[T.route != "blocked"]
    usage = pd.DataFrame([u for u in answered.usage if isinstance(u, dict)])
    if len(usage):
        print(f"\ntokens per answered request: prompt {usage.prompt_tokens.mean():.0f}, "
              f"completion {usage.completion_tokens.mean():.0f} (mean)")
    if "history_turns" in T and answered.history_turns.notna().any():
        print(f"history turns in the prompt: mean {answered.history_turns.mean():.2f}")


if __name__ == "__main__":
    main()
