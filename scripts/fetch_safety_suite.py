"""Collect public safety benchmarks into one test suite for the end-to-end guardrail evaluation.

Success = the agent's final answer is harmless and leaks nothing (a refusal, a safe redirect or a gentle answer all
count); harmful prompts measure the miss rate, benign look-alikes measure over-blocking. The suite is test-only:
never tune rules or thresholds on it.

    huggingface-cli login            # gated sets (SORRY-Bench, WildJailbreak) need the terms accepted on the Hub first
    python scripts/fetch_safety_suite.py                    # -> data/safety/suite.jsonl + data/safety/summary.json
    python scripts/fetch_safety_suite.py --only jbb,strongreject --semantic-dedup

Row: {id, source, label: harmful|benign, attack: direct|jailbreak|mutation:<style>, category, text}
Sources whose download fails are reported and skipped, so the rest still builds. Licenses differ per set (several
forbid redistribution): data/safety/ is local only.
"""
import argparse, csv, io, json, os, re, sys
from collections import Counter

TEXT = ["prompt", "adversarial", "behavior", "Behavior", "Goal", "goal", "forbidden_prompt", "question", "text",
        "query", "instruction", "turns"]
CAT = ["category", "Category", "harm_category", "risk_category", "SemanticCategory", "type", "focus"]


def pick(row, keys):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            v = row[k]
            return v[0] if isinstance(v, list) and v else v
    return None


def hf_rows(repo, config=None, split=None, **kw):
    from datasets import load_dataset
    d = load_dataset(repo, config, split=split, **kw) if config else load_dataset(repo, split=split, **kw)
    return list(d)


def hf_file(repo, path):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type="dataset")


# ------------------------------------------------------------------ sources (repo ids checked on the Hub, 2026-11)
def src_minorbench(a):
    for l in open("eval/minorbench.jsonl", encoding="utf-8"):       # from scripts/fetch_minorbench.py
        r = json.loads(l)
        yield dict(label="harmful", attack="direct", category=r.get("source_category", r["category"]), text=r["text"])


def src_jbb(a):                    # JailbreakBench/JBB-Behaviors, MIT: 100 harmful + 100 paired benign
    for split, label in (("harmful", "harmful"), ("benign", "benign")):
        for r in hf_rows("JailbreakBench/JBB-Behaviors", "behaviors", split):
            yield dict(label=label, attack="direct", category=pick(r, CAT), text=pick(r, TEXT))


def src_harmbench(a):              # walledai/HarmBench, MIT (standard behaviors)
    for r in hf_rows("walledai/HarmBench", "standard", "train"):
        yield dict(label="harmful", attack="direct", category=pick(r, CAT), text=pick(r, TEXT))


def src_strongreject(a):           # walledai/StrongREJECT, MIT: 313 prompts
    for r in hf_rows("walledai/StrongREJECT", split="train"):
        yield dict(label="harmful", attack="direct", category=pick(r, CAT), text=pick(r, TEXT))


def src_advbench(a):               # walledai/AdvBench: 520 prompts, overlaps the others (dedup handles it)
    for r in hf_rows("walledai/AdvBench", split="train"):
        yield dict(label="harmful", attack="direct", category=None, text=pick(r, TEXT))


def src_xstest(a):                 # XSTest: 250 safe look-alikes + 200 unsafe contrasts
    rows = hf_rows("walledai/XSTest", split="test")
    for r in rows:
        lab = str(r.get("label", r.get("type", ""))).lower()
        harmful = lab.startswith("unsafe") or lab.startswith("contrast")
        yield dict(label="harmful" if harmful else "benign", attack="direct", category=r.get("type"), text=pick(r, TEXT))


def src_wildjailbreak(a):          # allenai/wildjailbreak eval split (gated): adversarial harmful + benign
    rows = hf_rows("allenai/wildjailbreak", "eval", "train", delimiter="\t", keep_default_na=False)
    for r in rows:
        t = str(r.get("data_type", ""))
        if "harmful" in t or "benign" in t:
            yield dict(label="harmful" if "harmful" in t else "benign", attack="jailbreak" if "adversarial" in t
                       else "direct", category=t, text=r.get("adversarial") or r.get("vanilla"))


def src_sorrybench(a):             # sorry-bench/sorry-bench-202406 (gated): 440 base + 20 linguistic mutations
    repo = "sorry-bench/sorry-bench-202406"
    from huggingface_hub import list_repo_files
    files = [f for f in list_repo_files(repo, repo_type="dataset") if f.endswith((".csv", ".jsonl"))]
    for f in files:
        style = "direct" if f.endswith(".csv") or "base" in f else "mutation:" + re.sub(r"^question_|\.jsonl$", "",
                                                                                         os.path.basename(f))
        if style != "direct" and not a.sorry_mutations:
            continue
        path = hf_file(repo, f)
        rows = (list(csv.DictReader(open(path, encoding="utf-8"))) if f.endswith(".csv")
                else [json.loads(l) for l in open(path, encoding="utf-8")])
        for r in rows:
            yield dict(label="harmful", attack=style, category=pick(r, CAT), text=pick(r, TEXT))


def src_own_benign(a):             # the child questions of the v2 test split: the over-blocking set that matters most
    for l in open("data/children_sft_v2_splits/test.jsonl", encoding="utf-8"):
        yield dict(label="benign", attack="direct", category="child question", text=json.loads(l)["instruction"])


SOURCES = dict(minorbench=src_minorbench, jbb=src_jbb, harmbench=src_harmbench, strongreject=src_strongreject,
               advbench=src_advbench, sorrybench=src_sorrybench, wildjailbreak=src_wildjailbreak, xstest=src_xstest,
               own_benign=src_own_benign)
# Not on the Hub as of 2026-11: SproutBench (1,283 youth prompts, arXiv 2508.11009) and Safe-Child-LLM (200, GitHub
# The-Responsible-AI-Initiative); add them with --extra once downloaded: same row format, one JSON object per line.


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", t.lower())).strip()


def semantic_dedup(rows, thresh):
    from sentence_transformers import SentenceTransformer
    import numpy as np
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    for p in m.parameters():
        p.data = p.data.clone()                         # macOS misaligned-weights workaround
    keep = []
    for label in ("harmful", "benign"):
        sub = [r for r in rows if r["label"] == label and r["attack"] == "direct"]   # mutations are kept on purpose
        v = m.encode([r["text"] for r in sub], normalize_embeddings=True, batch_size=256, convert_to_numpy=True)
        kept_v = []
        for r, x in zip(sub, v):
            if kept_v and float(np.max(np.stack(kept_v) @ x)) >= thresh:
                continue
            kept_v.append(x); keep.append(r)
    keep += [r for r in rows if r["attack"] != "direct"]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of: " + ",".join(SOURCES))
    ap.add_argument("--out", default="data/safety")
    ap.add_argument("--sorry-mutations", action="store_true", help="also the 20 x 440 SORRY-Bench mutations")
    ap.add_argument("--semantic-dedup", action="store_true", help="drop direct prompts with cosine >= --dedup-thresh")
    ap.add_argument("--dedup-thresh", type=float, default=0.85)
    ap.add_argument("--extra", nargs="*", default=[], help="more jsonl files in the row format")
    a = ap.parse_args()
    names = a.only.split(",") if a.only else list(SOURCES)
    rows, report, seen = [], {}, set()
    for name in names:
        try:
            got = [dict(r, source=name) for r in SOURCES[name](a) if r.get("text")]
        except Exception as e:
            report[name] = f"FAILED: {type(e).__name__}: {str(e)[:160]}"; print(name, report[name]); continue
        new = []
        for r in got:
            key = (r["label"], r["attack"], norm(str(r["text"])))
            if key not in seen:
                seen.add(key); new.append(r)
        rows += new
        report[name] = dict(rows=len(got), after_exact_dedup=len(new), **Counter(r["label"] for r in new))
        print(name, report[name])
    for p in a.extra:
        rows += [dict(json.loads(l), source=os.path.basename(p)) for l in open(p, encoding="utf-8")]
    if a.semantic_dedup:
        n = len(rows); rows = semantic_dedup(rows, a.dedup_thresh); print(f"semantic dedup: {n} -> {len(rows)}")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "suite.jsonl"), "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            r["text"] = str(r["text"]); r["category"] = None if r.get("category") is None else str(r["category"])
            f.write(json.dumps(dict(id=f"{r['source']}-{i}", **r), ensure_ascii=False) + "\n")
    summary = dict(sources=report, total=len(rows), by_label=Counter(r["label"] for r in rows),
                   by_attack=Counter(r["attack"].split(":")[0] for r in rows))
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary["by_label"]), json.dumps(summary["by_attack"]))


if __name__ == "__main__":
    main()
