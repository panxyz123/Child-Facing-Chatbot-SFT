"""How well do the rule-based input guardrails catch unsafe messages, and how often do they fire on normal ones?

Unsafe and hard-negative cases: eval/guard_cases.jsonl (child-phrased messages per category, plus benign questions
that share words with unsafe ones, e.g. "Why do lions kill zebras?"). Normal traffic: every child message in the
v2 dataset (train, validation, test), which should almost never be stopped.

    python eval/guard_eval.py
    python eval/guard_eval.py --extra my_cases.jsonl     # more cases in the same format: {"category", "text"}
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sprout.guards import check_input  # noqa: E402

EXPECTED_ACTION = {"self_harm": "support", "personal_info": "redact"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="eval/guard_cases.jsonl")
    ap.add_argument("--extra", default=None)
    ap.add_argument("--splits", default="data/children_sft_v2_splits")
    ap.add_argument("--out", default="results/guardrails/guard_eval.json")
    a = ap.parse_args(argv)

    cases = [json.loads(l) for l in open(a.cases, encoding="utf-8")]
    if a.extra:
        cases += [json.loads(l) for l in open(a.extra, encoding="utf-8")]
    per_cat = collections.defaultdict(lambda: dict(n=0, caught=0, right_category=0, misses=[]))
    hard_neg = dict(n=0, false_positives=[])
    for c in cases:
        r = check_input(c["text"])
        if c["category"] == "benign_hard":
            hard_neg["n"] += 1
            if r.action != "allow":
                hard_neg["false_positives"].append(dict(text=c["text"], category=r.category))
            continue
        d = per_cat[c["category"]]
        d["n"] += 1
        caught = r.action != "allow"
        d["caught"] += caught
        d["right_category"] += r.category == c["category"]
        if not caught:
            d["misses"].append(c["text"])

    normal = []
    for split in ["train", "validation", "test"]:
        p = os.path.join(a.splits, f"{split}.jsonl")
        if os.path.exists(p):
            normal += [json.loads(l)["instruction"] for l in open(p, encoding="utf-8")]
    flagged = [(t, check_input(t)) for t in normal]
    flagged = [(t, r.category, r.action) for t, r in flagged if r.action != "allow"]

    n_unsafe = sum(d["n"] for d in per_cat.values())
    caught = sum(d["caught"] for d in per_cat.values())
    print(f"{'category':20s} {'n':>3s} {'caught':>7s} {'right cat':>9s}")
    for cat, d in per_cat.items():
        print(f"{cat:20s} {d['n']:3d} {d['caught'] / d['n']:7.0%} {d['right_category'] / d['n']:9.0%}")
    print(f"{'all unsafe':20s} {n_unsafe:3d} {caught / n_unsafe:7.0%}")
    print(f"\nhard negatives: {len(hard_neg['false_positives'])}/{hard_neg['n']} wrongly stopped")
    print(f"normal child messages (v2 dataset): {len(flagged)}/{len(normal)} stopped or redacted "
          f"({len(flagged) / max(1, len(normal)):.1%})")
    print("\nmissed unsafe messages:")
    for cat, d in per_cat.items():
        for m in d["misses"]:
            print(f"  [{cat}] {m}")
    if hard_neg["false_positives"] or flagged:
        print("\nfalse positives:")
        for fp in hard_neg["false_positives"]:
            print(f"  [hard negative -> {fp['category']}] {fp['text']}")
        for t, cat, act in flagged[:30]:
            print(f"  [dataset -> {cat}/{act}] {t[:100]}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(unsafe_recall=caught / n_unsafe, per_category={k: {x: v for x, v in d.items()} for k, d in per_cat.items()},
                   hard_negatives=hard_neg, normal_flag_rate=len(flagged) / max(1, len(normal)),
                   normal_flagged=[dict(text=t, category=c, action=x) for t, c, x in flagged]),
              open(a.out, "w"), indent=2)
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
