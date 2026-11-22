"""Article-weighted retrieval on paragraph chunks (port of the earlier project's select_best_files_and_rerank),
evaluated against the existing methods on the same queries, judged with the same incremental Gemini judge.

  recall   hybrid (alpha 0.5) over wiki_v4_para chunks, top RECALL
  rerank   cross-encoder on those chunks; scores -> sigmoid, so they are positive like the original reranker's
  article  score = sum over its recalled chunks of sigmoid(ce) * 1 / (ce rank + 1); keep the top N_ARTICLES
           articles whose score >= THRESH * best
  chunks   the top TOP_CHUNKS chunks (by ce) of the kept articles; adjacent chunks of one article are merged
           (sentences already present are dropped, since chunks overlap by one paragraph)
  ablation para_rerank: the same recall + cross-encoder, no article step (shows what the article step adds)

    python rag/article_weighted.py --n 60                       # -> results/retrieval_v4/{runs,relevance}.jsonl
    python rag/retrieval_benchmark.py --stage judge  --out results/retrieval_v4          # new passages only
    python rag/retrieval_benchmark.py --stage report --out results/retrieval_v4 --router models/router.joblib
"""
import argparse, csv, json, math, os, random, re, shutil, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrieval_benchmark as RB

RECALL, N_ARTICLES, THRESH, TOP_CHUNKS = 20, 2, 0.8, 3      # = RETRIEVAL_K, top-2 files, FILE_SCORE_THRESH, TOP_N
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


class ParaRetriever(RB.Retriever):
    def __init__(self, db_path, table_name):
        super().__init__(db_path, table_name)
        self.cols = ["id", "article_id", "topic", "para_idx", "search_text"]

    @staticmethod
    def _fuse(vec, fts, alpha):
        import pandas as pd
        keep = ["id", "article_id", "topic", "para_idx", "search_text"]
        m = pd.merge(vec[keep + ["score"]], fts[keep + ["score"]], on="id", how="outer", suffixes=("_v", "_t"))
        for c in keep[1:]:
            m[c] = m[c + "_v"].fillna(m[c + "_t"])
        m["score"] = (1 - alpha) * RB.minmax(m.score_v.fillna(0.0)) + alpha * RB.minmax(m.score_t.fillna(0.0))
        return m.sort_values("score", ascending=False)[keep + ["score"]]

    def run(self, q):
        t0 = time.perf_counter()
        vec, _ = self._vector(q, RB.CANDIDATES)
        fts, _ = self._fts(q, RB.CANDIDATES)
        pool = self._fuse(vec, fts, 0.5).head(RECALL).copy()
        pool["ce"] = self.reranker.predict([(q, t) for t in pool.search_text.tolist()])
        pool["s"] = 1 / (1 + np.exp(-pool.ce))
        pool = pool.sort_values("ce", ascending=False).reset_index(drop=True)
        t_rerank = time.perf_counter() - t0
        para = [dict(id=r.id, text=r.search_text, score=float(r.ce)) for r in pool.head(RB.TOP_K).itertuples()]

        art = {}
        for rank, r in enumerate(pool.itertuples()):
            art[r.article_id] = art.get(r.article_id, 0.0) + r.s / (rank + 1)
        ranked = sorted(art.items(), key=lambda x: -x[1])
        best = ranked[0][1]
        keep = [a for a, sc in ranked[:N_ARTICLES] if sc >= THRESH * best]
        top = pool[pool.article_id.isin(keep)].head(TOP_CHUNKS)
        spans = []
        for a in keep:                                      # merge adjacent chunks of one article
            rows = top[top.article_id == a].sort_values("para_idx")
            groups = []
            for r in rows.itertuples():
                if groups and r.para_idx - groups[-1][-1].para_idx <= 1:
                    groups[-1].append(r)
                else:
                    groups.append([r])
            for g in groups:
                seen, out = set(), []
                for k, r in enumerate(g):
                    body = r.search_text if k == 0 else r.search_text.split(": ", 1)[-1]
                    for s in SENT.split(body):
                        if s not in seen:
                            seen.add(s); out.append(s)
                ids = [int(r.para_idx) for r in g]
                spans.append(dict(id=f"{a}#p{ids[0]}" + (f"-{ids[-1]}" if len(ids) > 1 else ""),
                                  text=" ".join(out), score=float(max(r.ce for r in g))))
        spans.sort(key=lambda d: -d["score"])
        return {"para_rerank": (para, t_rerank), "article_weighted": (spans, time.perf_counter() - t0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="queries, sampled from the v3 router-routed set (0 = all)")
    ap.add_argument("--db", default="rag/lancedb_data")
    ap.add_argument("--table", default="wiki_v4_para")
    ap.add_argument("--base", default="results/retrieval_v3", help="runs + grades of the existing methods")
    ap.add_argument("--out", default="results/retrieval_v4")
    a = ap.parse_args()
    qids = [r["qid"] for r in csv.DictReader(open(os.path.join(a.base, "per_query.csv"), encoding="utf-8"))]
    if a.n and a.n < len(qids):
        qids = sorted(random.Random(7).sample(qids, a.n), key=lambda q: int(q.split("-")[1]))
    base = {json.loads(l)["qid"]: json.loads(l) for l in open(os.path.join(a.base, "runs.jsonl"), encoding="utf-8")}
    queries = {q["qid"]: q for q in RB.load_queries("data/children_sft_v2_splits", "test")}
    os.makedirs(a.out, exist_ok=True)
    rel_dst = os.path.join(a.out, "relevance.jsonl")
    if not os.path.exists(rel_dst):                          # reuse every grade already paid for
        shutil.copy(os.path.join(a.base, "relevance.jsonl"), rel_dst)
    path = os.path.join(a.out, "runs.jsonl")
    done = {json.loads(l)["qid"] for l in open(path)} if os.path.exists(path) else set()
    todo = [q for q in qids if q not in done]
    print(f"{len(qids)} queries, {len(todo)} to run; table {a.table}")
    if not todo:
        return
    r = ParaRetriever(a.db, a.table); r.run("warm up the models")
    with open(path, "a", encoding="utf-8") as f:
        for i, qid in enumerate(todo, 1):
            rec = base[qid]
            for m, (items, t) in r.run(queries[qid]["question"]).items():
                rec["methods"][m] = dict(seconds=round(t, 4), ids=[d["id"] for d in items],
                                         scores=[d["score"] for d in items],
                                         passages={d["id"]: d["text"][:1200] for d in items})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")
    print("note: article_weighted returns at most", TOP_CHUNKS, "spans, so its @5 metrics use fewer items")


if __name__ == "__main__":
    main()
