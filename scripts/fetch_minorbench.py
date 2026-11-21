"""Download MinorBench (Khoo et al., ICLR 2025 AI4CHL workshop; 299 child-phrased unsafe queries in 6 categories;
MIT license) and convert it to the guardrail eval format, as a held-out test set the rules were not written against.

    python scripts/fetch_minorbench.py
    python eval/guard_eval.py --cases eval/minorbench.jsonl --out results/guardrails/minorbench.json

Category mapping to the guardrail categories: Danger -> dangerous_activity, Sexual -> sexual, Self-Harm -> self_harm,
Substance Use -> drugs_alcohol. Profanities and Hateful have no rules yet and are kept as profanity / hateful.
"""
import argparse
import csv
import io
import json
import os
import urllib.request

REPO, FILE = "govtech/MinorBench", "test.csv"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/{FILE}"
CATEGORY_MAP = {"danger": "dangerous_activity", "sexual": "sexual", "self-harm": "self_harm", "self harm": "self_harm",
                "selfharm": "self_harm", "substance use": "drugs_alcohol", "substance": "drugs_alcohol",
                "profanities": "profanity", "profanity": "profanity", "hateful": "hateful", "hate": "hateful"}
TEXT_COLS = ["prompt", "query", "text", "question", "input", "user_query"]
CAT_COLS = ["category", "risk_category", "risk", "label", "type"]


def download():
    try:
        from huggingface_hub import hf_hub_download
        return open(hf_hub_download(REPO, FILE, repo_type="dataset"), encoding="utf-8").read()
    except Exception as e:
        print(f"huggingface_hub download failed ({str(e)[:80]}), trying {URL}")
        return urllib.request.urlopen(URL, timeout=60).read().decode("utf-8")


def pick(cols, wanted, given):
    if given:
        return given
    low = {c.lower().strip(): c for c in cols}
    return next((low[w] for w in wanted if w in low), None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/minorbench.jsonl")
    ap.add_argument("--text-col", default=None, help="override if the column is not detected")
    ap.add_argument("--cat-col", default=None)
    a = ap.parse_args(argv)

    rows = list(csv.DictReader(io.StringIO(download())))
    cols = list(rows[0].keys())
    text_col, cat_col = pick(cols, TEXT_COLS, a.text_col), pick(cols, CAT_COLS, a.cat_col)
    print("columns:", cols)
    if not text_col or not cat_col:
        raise SystemExit(f"could not detect the text / category columns; pass --text-col and --cat-col (columns: {cols})")
    counts, unknown = {}, set()
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            raw = (r[cat_col] or "").strip()
            cat = CATEGORY_MAP.get(raw.lower())
            if cat is None:
                unknown.add(raw); cat = raw.lower().replace(" ", "_") or "unknown"
            text = (r[text_col] or "").strip()
            if text:
                f.write(json.dumps(dict(category=cat, text=text, source="MinorBench", source_category=raw)) + "\n")
                counts[cat] = counts.get(cat, 0) + 1
    print(f"wrote {sum(counts.values())} cases to {a.out}: {counts}")
    if unknown:
        print("categories kept as-is (not in the mapping):", sorted(unknown))


if __name__ == "__main__":
    main()
