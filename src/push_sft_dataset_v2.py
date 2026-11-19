"""
Split the v2 SFT dataset and push it to the Hugging Face Hub as train / validation / test.

Split: stratified by age, and inside each age by needs_knowledge, so every split has the same age mix and the
same share of knowledge questions per age (the test split is also the RAG / hallucination eval set).
    per age group: TRAIN_PER_AGE train, VAL_PER_AGE validation, the rest test
Duplicates:
    - same instruction, same age (repeated child utterances): keep one row
    - same instruction, different ages (the HF source asks some questions at several ages): keep all of them,
      since they teach "answer for the set age", but put the whole group in train, swapping in other rows of the
      same age and knowledge type, so validation / test never contain a question seen in training

Run from the repo root (the folder with data/ and src/):
    python src/push_sft_dataset_v2.py                 # dry run: print split stats, write local copies
    python src/push_sft_dataset_v2.py --push          # also push to the Hub (needs `huggingface-cli login` or HF_TOKEN)
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

IN_PATH = "data/children_sft_v2.jsonl"
OUT_DIR = "data/children_sft_v2_splits"
REPO_ID = "yxpan/children_sft_dataset_v2"
TRAIN_PER_AGE, VAL_PER_AGE = 140, 20
SEED = 42
AGES = ["0-3", "4-6", "7-9", "10-13", "14-18"]
COLUMNS = ["id", "age", "intent", "needs_knowledge", "instruction", "response",
           "source", "age_source", "origin", "fkgl"]


def allocate(sizes, total):
    """Split `total` across strata in proportion to `sizes` (largest remainder), so counts sum exactly."""
    n = sum(sizes.values())
    raw = {k: total * v / n for k, v in sizes.items()}
    out = {k: int(r) for k, r in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: total - sum(out.values())]:
        out[k] += 1
    return out


def norm(text):
    return " ".join(str(text).lower().split())


def drop_same_age_duplicates(rows):
    seen, kept = set(), []
    for r in rows:
        key = (norm(r["instruction"]), r["age"])
        if key not in seen:
            seen.add(key)
            kept.append(r)
    return kept


def keep_groups_in_train(splits, rng):
    """Move every cross-age duplicate group fully into train, swapping each moved row with a train row of the
    same age and needs_knowledge that is not part of any group, so split sizes and stratification stay the same."""
    count = Counter(norm(r["instruction"]) for rows in splits.values() for r in rows)
    grouped = {k for k, v in count.items() if v > 1}
    moved = 0
    for name in ("validation", "test"):
        for i, r in enumerate(splits[name]):
            if norm(r["instruction"]) not in grouped:
                continue
            candidates = [j for j, t in enumerate(splits["train"])
                          if t["age"] == r["age"] and t["needs_knowledge"] == r["needs_knowledge"]
                          and norm(t["instruction"]) not in grouped]
            j = rng.choice(candidates)
            splits[name][i], splits["train"][j] = splits["train"][j], r
            moved += 1
    return len(grouped), moved


def split_rows(rows):
    rng = random.Random(SEED)
    splits = {"train": [], "validation": [], "test": []}
    for age in AGES:
        group = [r for r in rows if r["age"] == age]
        strata = defaultdict(list)
        for r in group:
            strata[r["needs_knowledge"]].append(r)
        for s in strata.values():
            rng.shuffle(s)
        sizes = {k: len(v) for k, v in strata.items()}
        n_train, n_val = allocate(sizes, TRAIN_PER_AGE), allocate(sizes, VAL_PER_AGE)
        for k, s in strata.items():
            splits["train"] += s[:n_train[k]]
            splits["validation"] += s[n_train[k]:n_train[k] + n_val[k]]
            splits["test"] += s[n_train[k] + n_val[k]:]
    n_groups, moved = keep_groups_in_train(splits, rng)
    print(f"cross-age duplicate groups: {n_groups}, rows swapped into train: {moved}")
    for name in splits:
        rng.shuffle(splits[name])
    return splits


def summarize(splits):
    for name, rows in splits.items():
        ages = Counter(r["age"] for r in rows)
        know = {a: sum(1 for r in rows if r["age"] == a and r["needs_knowledge"]) for a in AGES}
        print(f"{name:>10}: {len(rows):4d} rows | " +
              "  ".join(f"{a}: {ages.get(a, 0)} ({know[a]} knowledge)" for a in AGES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="push to the Hugging Face Hub")
    ap.add_argument("--private", action="store_true", help="create the Hub dataset as private")
    ap.add_argument("--repo", default=REPO_ID)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(IN_PATH, encoding="utf-8")]
    n0 = len(rows)
    rows = drop_same_age_duplicates(rows)
    print(f"rows: {n0}, after dropping same-age duplicates: {len(rows)}")
    for i, r in enumerate(rows):
        r["id"] = i
    assert all(r["needs_knowledge"] is not None for r in rows), "some rows have no needs_knowledge label"

    splits = split_rows(rows)
    assert sum(len(v) for v in splits.values()) == len(rows)
    ids = [r["id"] for v in splits.values() for r in v]
    assert len(set(ids)) == len(ids), "a row landed in two splits"
    train_q = {norm(r["instruction"]) for r in splits["train"]}
    leaks = [r for name in ("validation", "test") for r in splits[name] if norm(r["instruction"]) in train_q]
    assert not leaks, f"{len(leaks)} validation/test questions also appear in train"
    summarize(splits)

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, rows_ in splits.items():
        with open(os.path.join(OUT_DIR, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows_:
                f.write(json.dumps({c: r.get(c) for c in COLUMNS}, ensure_ascii=False) + "\n")
    print(f"local copies -> {OUT_DIR}/")

    if not args.push:
        print("dry run: nothing pushed. Re-run with --push to upload.")
        return
    from datasets import Dataset, DatasetDict
    dd = DatasetDict({name: Dataset.from_list([{c: r.get(c) for c in COLUMNS} for r in rows_])
                      for name, rows_ in splits.items()})
    dd.push_to_hub(args.repo, private=args.private)
    print(f"pushed -> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
