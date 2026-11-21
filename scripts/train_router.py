"""Train the knowledge router on the v2 dataset (train + validation), report on the test split, save to models/.

    python scripts/train_router.py                       # MiniLM sentence embeddings + logistic regression
    python scripts/train_router.py --features tfidf      # no model download; about 87% test accuracy

The test report includes accuracy per age group and recall on knowledge questions at the agent's threshold,
since a missed knowledge question goes to the model without retrieval.
"""
import argparse
import json
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

AGES = ["0-3", "4-6", "7-9", "10-13", "14-18"]


def load(splits, name):
    return [json.loads(l) for l in open(os.path.join(splits, f"{name}.jsonl"), encoding="utf-8")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/children_sft_v2_splits")
    ap.add_argument("--features", default="sentence-transformers/all-MiniLM-L6-v2", help="a sentence-transformers model or 'tfidf'")
    ap.add_argument("--threshold", type=float, default=0.4, help="the agent's knowledge_threshold")
    ap.add_argument("--out", default="models/router.joblib")
    a = ap.parse_args(argv)

    train = load(a.splits, "train") + load(a.splits, "validation")
    test = load(a.splits, "test")
    X = lambda rows: [r["instruction"] for r in rows]
    y = lambda rows: np.array([int(bool(r["needs_knowledge"])) for r in rows])

    bundle = dict(features=a.features)
    if a.features == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(X(train))
        enc = vec.transform
        bundle["vectorizer"] = vec
    else:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(a.features, device="cpu")
        enc = lambda texts: st.encode(texts, normalize_embeddings=True, batch_size=64)

    clf = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced").fit(enc(X(train)), y(train))
    p = clf.predict_proba(enc(X(test)))[:, 1]
    yt = y(test)
    report = {}
    for name, t in [("0.5", 0.5), (f"{a.threshold} (agent)", a.threshold)]:
        pred = (p >= t).astype(int)
        report[name] = dict(accuracy=accuracy_score(yt, pred), f1=f1_score(yt, pred),
                            knowledge_recall=recall_score(yt, pred), knowledge_precision=precision_score(yt, pred),
                            retrieval_rate=float(pred.mean()))
    pred = (p >= a.threshold).astype(int)
    report["accuracy_by_age"] = {g: float((pred == yt)[[r["age"] == g for r in test]].mean()) for g in AGES}
    print(json.dumps(report, indent=2))
    misses = [(r["age"], r["instruction"]) for r, pi, yi in zip(test, pred, yt) if yi == 1 and pi == 0]
    print(f"\nknowledge questions sent without retrieval at threshold {a.threshold}: {len(misses)}")
    for age, q in misses[:10]:
        print(f"  [{age}] {q[:100]}")

    bundle.update(clf=clf, meta=dict(features=a.features, n_train=len(train), test_report=report))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    joblib.dump(bundle, a.out)
    json.dump(report, open(os.path.splitext(a.out)[0] + "_report.json", "w"), indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
