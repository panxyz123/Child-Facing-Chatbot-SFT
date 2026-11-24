"""Input-guard baseline on the public safety suite (scripts/fetch_safety_suite.py): how many harmful prompts the
input guard stops before the model sees them, and how many benign ones it stops by mistake. This is the first layer
only: a harmful prompt that passes it can still get a safe answer from the model and the output guard, which the
end-to-end evaluation measures.

    python eval/guard_suite_eval.py                       # -> results/guardrails/suite_input.json (+ printed tables)

Counted as stopped: any action other than allow (block, support, redact). Wilson 95% intervals.
"""
import argparse, collections, json, math, os, sys
sys.path.insert(0, os.getcwd())
from sprout.guards import check_input


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def table(rows, key, label):
    g = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        g[r[key]][0] += r["stopped"]; g[r[key]][1] += 1
    print(f"\n{label} by {key}")
    out = {}
    for k, (s, n) in sorted(g.items(), key=lambda x: -x[1][1]):
        lo, hi = wilson(s, n); out[k] = dict(n=n, stopped=s, rate=s / n, ci95=[lo, hi])
        print(f"  {str(k)[:38]:38s} n={n:5d}  stopped {s / n:6.1%}  [{lo:.1%}, {hi:.1%}]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="data/safety/suite.jsonl")
    ap.add_argument("--out", default="results/guardrails/suite_input.json")
    ap.add_argument("--misses", type=int, default=15, help="harmful prompts that passed, printed per source")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.suite, encoding="utf-8")]
    for r in rows:
        g = check_input(r["text"]); r["stopped"] = g.action != "allow"; r["guard_category"] = g.category
        r["attack"] = r["attack"].split(":")[0]
    harm = [r for r in rows if r["label"] == "harmful"]; ben = [r for r in rows if r["label"] == "benign"]
    hs, bs = sum(r["stopped"] for r in harm), sum(r["stopped"] for r in ben)
    print(f"harmful stopped: {hs}/{len(harm)} = {hs / max(len(harm), 1):.1%}  95% CI {wilson(hs, len(harm))}")
    print(f"benign stopped (over-blocking): {bs}/{len(ben)} = {bs / max(len(ben), 1):.1%}  95% CI {wilson(bs, len(ben))}")
    res = dict(harmful=dict(n=len(harm), stopped=hs, by_source=table(harm, "source", "harmful"),
                            by_attack=table(harm, "attack", "harmful")),
               benign=dict(n=len(ben), stopped=bs, by_source=table(ben, "source", "benign")))
    print("\nexamples of harmful prompts that passed the input guard:")
    by = collections.defaultdict(list)
    for r in harm:
        if not r["stopped"]:
            by[r["source"]].append(r["text"])
    for s, xs in by.items():
        for t in xs[: max(1, a.misses // max(len(by), 1))]:
            print(f"  [{s}] {t[:140]}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    with open(a.out.replace(".json", "_rows.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "source", "label", "attack", "stopped", "guard_category")}) + "\n")
    print("saved", a.out)


if __name__ == "__main__":
    main()
