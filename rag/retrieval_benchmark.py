"""Retrieval benchmark on real child questions: can the Wikipedia science index find what a knowledge question needs?

Queries are the knowledge questions of the v2 test split, as the child asked them (no rewriting). For each query,
every method returns its top 10; the top 5 of all methods are pooled and Gemini grades each pooled passage
(2 = contains what the answer needs, 1 = on topic but not enough, 0 = not relevant). Metrics per method:

  success@k   share of queries with a grade-2 passage in the top k (k = 1, 3, 5): the answer is in the context
  MRR         mean reciprocal rank of the first grade-2 passage
  nDCG@5      graded ranking quality
  latency     p50 / p95 per query, including query embedding (CPU)
  coverage    share of queries where any method found a grade-2 passage: the upper bound the corpus allows at
              this pool depth, split by the question's domain (from the hallucination judge, when available)
  gate        for hybrid + rerank: how well the top-1 reranker score predicts "the top passage answers the
              question" (AUROC, and precision / recall at thresholds): the relevance gate of the agent

Stages (resumable; each caches its results under --out):
  retrieve   run all methods, store top-10 ids, passages, scores and timings      (local CPU, no API)
  judge      grade the pooled passages with Gemini, one call per query               (stops at the daily quota)
  report     compute the metrics above from the two caches

    python rag/retrieval_benchmark.py --stage retrieve --db rag/lancedb_data
    python rag/retrieval_benchmark.py --stage judge --limit 3     # check a few first
    python rag/retrieval_benchmark.py --stage judge
    python rag/retrieval_benchmark.py --stage report

Needs: lancedb, sentence-transformers, pandas, scikit-learn, google-genai, python-dotenv, pydantic.
"""
import argparse
import json
import os
import re
import sys
import time
from typing import List, Literal

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"     # same model the index was built with
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 100          # per retriever before fusion / reranking
TOP_K = 10                # stored per method
POOL_DEPTH = 5            # top passages per method sent to the judge
RERANK_POOL = 50          # hybrid candidates the cross-encoder reorders
ALPHAS = [0.3, 0.5, 0.7]  # weight on the full-text score in hybrid fusion (1 - alpha on the vector score)
SCIENCE = {"physical science", "life science", "earth and space"}

JUDGE_PROMPT = """A child asked a question. Below are passages retrieved from an encyclopedia. For each passage,
grade whether it helps answer the question:
  2 = contains the information needed to answer the question correctly (even if written for adults)
  1 = on the right topic, but does not contain what the answer needs
  0 = not relevant
Judge each passage on its own. Return one grade per passage id.

Question: {question}

Passages:
{passages}
"""


# ----------------------------------------------------------------------------- retrieval
def fts_query(text):
    """Full-text query syntax breaks on quotes and punctuation in child questions: keep words only."""
    return " ".join(re.findall(r"[A-Za-z0-9]+", text.replace("’", "'"))) or "a"


def minmax(s):
    return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0.0


class Retriever:
    def __init__(self, db_path, table_name):
        import lancedb
        from sentence_transformers import CrossEncoder, SentenceTransformer
        self.table = lancedb.connect(db_path).open_table(table_name)
        self.embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        self.reranker = CrossEncoder(RERANK_MODEL, device="cpu")
        # transformers 5.23 + torch 2.15 on macOS CPU: weights loaded as-is give NaN in float32 linear layers;
        # cloning them fixes it
        for m in (self.embedder, self.reranker.model):
            for p in m.parameters():
                p.data = p.data.clone()
        self.cols = ["id", "topic", "search_text"]

    def _vector(self, q, k):
        t0 = time.perf_counter()
        v = self.embedder.encode(q).tolist()
        df = self.table.search(v).metric("cosine").select(self.cols).limit(k).to_pandas()
        df["score"] = 1.0 - df["_distance"]
        return df, time.perf_counter() - t0

    def _fts(self, q, k):
        t0 = time.perf_counter()
        df = self.table.search(fts_query(q), query_type="fts").select(self.cols).limit(k).to_pandas()
        df["score"] = df["_score"] if "_score" in df else 0.0
        return df, time.perf_counter() - t0

    @staticmethod
    def _fuse(vec, fts, alpha):
        m = pd.merge(vec[["id", "topic", "search_text", "score"]], fts[["id", "topic", "search_text", "score"]],
                     on="id", how="outer", suffixes=("_v", "_t"))
        m["topic"] = m.topic_v.fillna(m.topic_t); m["search_text"] = m.search_text_v.fillna(m.search_text_t)
        m["score"] = (1 - alpha) * minmax(m.score_v.fillna(0.0)) + alpha * minmax(m.score_t.fillna(0.0))
        return m.sort_values("score", ascending=False)[["id", "topic", "search_text", "score"]]

    def run(self, q):
        """All methods for one query: {method: (top-k DataFrame, seconds)}."""
        out = {}
        vec, t_vec = self._vector(q, CANDIDATES)
        fts, t_fts = self._fts(q, CANDIDATES)
        out["vector"] = (vec.head(TOP_K), t_vec)
        out["fts"] = (fts.head(TOP_K), t_fts)
        for a in ALPHAS:
            t0 = time.perf_counter(); fused = self._fuse(vec, fts, a); t_fuse = time.perf_counter() - t0
            out[f"hybrid_a{a}"] = (fused.head(TOP_K), t_vec + t_fts + t_fuse)
        t0 = time.perf_counter()
        pool = self._fuse(vec, fts, 0.5).head(RERANK_POOL).copy()
        pool["score"] = self.reranker.predict([(q, t) for t in pool.search_text.tolist()])
        pool = pool.sort_values("score", ascending=False)
        out["hybrid_rerank"] = (pool.head(TOP_K), t_vec + t_fts + time.perf_counter() - t0)
        return out


def load_queries(splits_dir, split):
    rows = []
    for i, l in enumerate(open(os.path.join(splits_dir, f"{split}.jsonl"), encoding="utf-8")):
        r = json.loads(l)
        if r["needs_knowledge"]:
            rows.append(dict(qid=f"{split}-{i}", age=r["age"], question=r["instruction"], reference=r["response"]))
    return rows


def stage_retrieve(a, queries):
    path = os.path.join(a.out, "runs.jsonl")
    done = {json.loads(l)["qid"] for l in open(path)} if os.path.exists(path) else set()
    pending = [q for q in queries if q["qid"] not in done]
    print(f"retrieve: {len(queries)} queries, {len(pending)} to go")
    if not pending:
        return
    r = Retriever(a.db, a.table)
    r.run("warm up the models")
    with open(path, "a", encoding="utf-8") as f:
        for i, q in enumerate(pending, 1):
            res = r.run(q["question"])
            rec = dict(qid=q["qid"], methods={
                m: dict(seconds=round(t, 4), ids=df.id.astype(str).tolist(), scores=[float(x) for x in df.score],
                        passages={str(d): str(txt)[:1200] for d, txt in zip(df.id, df.search_text)})
                for m, (df, t) in res.items()})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 20 == 0 or i == len(pending):
                print(f"  {i}/{len(pending)}")


# ----------------------------------------------------------------------------- judging
def make_relevance_judge(client, model_id):
    from pydantic import BaseModel

    class Grade(BaseModel):
        id: str
        grade: int                      # 0 / 1 / 2; Gemini schemas reject integer enums, so checked below

    class Grades(BaseModel):            # wrap the list in an object: a top-level typing.List is not accepted
        grades: List[Grade]

    def judge(question, passages):
        listing = "\n".join(f"[{pid}] {text}" for pid, text in passages.items())
        out = client.models.generate_content(
            model=model_id, contents=JUDGE_PROMPT.format(question=question, passages=listing),
            config={"response_mime_type": "application/json", "response_schema": Grades})
        return {g.id: int(g.grade) for g in out.parsed.grades if g.grade in (0, 1, 2)}
    return judge


def pooled_passages(run):
    pool = {}
    for m in run["methods"].values():
        for d in m["ids"][:POOL_DEPTH]:
            pool[d] = m["passages"][d]
    return pool


def load_relevance(path):
    """qid -> {passage id: grade}. A query can have several records (later runs add the passages that earlier
    runs had not seen), so grades are merged, not overwritten."""
    rel = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            rel.setdefault(r["qid"], {}).update(r["grades"])
    return rel


def stage_judge(a, queries):
    """Incremental: for every query, grade only the pooled passages that have no grade yet. Re-running after a
    method changed (or after a call left some passages out) costs one call per query that has new passages."""
    from judge_hallucination import backoff, check_fatal, make_client
    runs = {json.loads(l)["qid"]: json.loads(l) for l in open(os.path.join(a.out, "runs.jsonl"))}
    path = os.path.join(a.out, "relevance.jsonl")
    rel = load_relevance(path)
    todo = []
    for q in queries:
        if q["qid"] not in runs:
            continue
        pool = pooled_passages(runs[q["qid"]])
        new = {d: t for d, t in pool.items() if d not in rel.get(q["qid"], {})}
        if new:
            todo.append((q, new))
    todo = todo[: a.limit]
    print(f"judge: {len(todo)} queries have passages without a grade ({sum(len(n) for _, n in todo)} passages)")
    judge = make_relevance_judge(make_client(), a.model)
    failed = 0
    for i, (q, new) in enumerate(todo, 1):
        for attempt in range(4):
            try:
                grades = judge(q["question"], new)
                grades = {d: g for d, g in grades.items() if d in new}      # ignore ids the model made up
                missing = sorted(set(new) - set(grades))
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(qid=q["qid"], grades=grades, missing=missing, judge_model=a.model)) + "\n")
                n2 = sum(v == 2 for v in grades.values())
                print(f"[{i}/{len(todo)}] {len(new)} new passages, {n2} answer it | {q['question'][:60]}")
                failed = 0
                break
            except Exception as e:
                check_fatal(e)
                wait = backoff(e, attempt)
                print(f"  attempt {attempt + 1} failed ({str(e)[:80]}), sleeping {wait}s")
                time.sleep(wait)
        else:
            failed += 1
            if failed >= 3:
                raise SystemExit("Stopping: 3 queries in a row failed. Re-run later; progress is saved.")
        time.sleep(2)


# ----------------------------------------------------------------------------- report
def dcg(gains):
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))


def stage_report(a, queries):
    from sklearn.metrics import roc_auc_score
    runs = {json.loads(l)["qid"]: json.loads(l) for l in open(os.path.join(a.out, "runs.jsonl"))}
    rel = load_relevance(os.path.join(a.out, "relevance.jsonl"))
    Q = pd.DataFrame([q for q in queries if q["qid"] in rel and q["qid"] in runs]).set_index("qid")
    if a.router:
        # keep only the questions the agent would actually send to retrieval
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from sprout.router import KnowledgeRouter
        router = KnowledgeRouter(a.router)
        Q["p_knowledge"] = [router.p_knowledge(t) for t in Q.question]
        dropped = Q[Q.p_knowledge < a.router_threshold]
        print(f"router (threshold {a.router_threshold}): {len(Q) - len(dropped)} of {len(Q)} queries are retrieved; "
              f"dropped, for example: {dropped.question.head(6).tolist()}")
        Q = Q[Q.p_knowledge >= a.router_threshold]
    print(f"report on {len(Q)} graded queries")
    domain = {}
    jpath = a.domains
    if jpath and os.path.exists(jpath):
        domain = {json.loads(l)["id"]: json.loads(l)["domain"] for l in open(jpath)}
    Q["domain"] = [domain.get(q, "unknown") for q in Q.index]
    Q["science"] = Q.domain.map(lambda d: "science" if d in SCIENCE else ("unknown" if d == "unknown" else "non-science"))

    methods = list(next(iter(runs.values()))["methods"])
    partial = [m for m in methods if any(m not in runs[q]["methods"] for q in Q.index)]
    if partial:                         # e.g. a method still running: report the rest instead of failing
        print(f"skipping methods missing for some queries: {partial}")
        methods = [m for m in methods if m not in partial]
    rows, per_query = [], {}
    for m in methods:
        s1 = s3 = s5 = rr = nd = 0.0; secs = []
        hits = []
        for qid in Q.index:
            g = rel[qid]; r = runs[qid]["methods"][m]
            gains = [g.get(d, 0) for d in r["ids"][:POOL_DEPTH]]
            first = next((i for i, x in enumerate(gains) if x == 2), None)
            s1 += first is not None and first < 1; s3 += first is not None and first < 3; s5 += first is not None
            rr += 1 / (first + 1) if first is not None else 0
            ideal = sorted([v for v in g.values()], reverse=True)[:POOL_DEPTH]
            nd += dcg(gains) / dcg(ideal) if dcg(ideal) > 0 else 0
            secs.append(r["seconds"]); hits.append(first is not None)
        n = len(Q)
        per_query[m] = hits
        rows.append(dict(method=m, success_at_1=s1 / n, success_at_3=s3 / n, success_at_5=s5 / n, mrr=rr / n,
                         ndcg_at_5=nd / n, p50_ms=1000 * np.percentile(secs, 50), p95_ms=1000 * np.percentile(secs, 95)))
    M = pd.DataFrame(rows).set_index("method")
    M["judged_at_5"] = [np.mean([d in rel[q] for q in Q.index for d in runs[q]["methods"][m]["ids"][:POOL_DEPTH]])
                        for m in M.index]       # below 1.0: ungraded passages count as 0, run --stage judge first
    pd.set_option("display.width", 160)
    print("\nRetrieval quality and latency (CPU)\n" + M.round(3).to_string())

    Q["covered"] = [any(v == 2 for v in rel[q].values()) for q in Q.index]
    best = M.success_at_5.idxmax()
    Q["best_method_hit"] = per_query[best]
    cov = Q.groupby("science").agg(n=("covered", "size"), corpus_has_answer=("covered", "mean"),
                                   **{f"{best}_success_at_5": ("best_method_hit", "mean")})
    print(f"\nCoverage: any method found a passage that answers the question (pool depth {POOL_DEPTH})")
    print(f"  all queries: {Q.covered.mean():.1%}")
    print(cov.round(3).to_string())
    by_age = Q.groupby("age").agg(n=("covered", "size"), corpus_has_answer=("covered", "mean"),
                                  **{f"{best}_success_at_5": ("best_method_hit", "mean")})
    by_age = by_age.reindex([x for x in ["0-3", "4-6", "7-9", "10-13", "14-18"] if x in by_age.index])
    print("\nBy age (child phrasing gets simpler and vaguer for younger ages)")
    print(by_age.round(3).to_string())

    # relevance gate on the reranker's top-1 score
    top_score = np.array([runs[q]["methods"]["hybrid_rerank"]["scores"][0] for q in Q.index])
    top_ok = np.array([rel[q].get(runs[q]["methods"]["hybrid_rerank"]["ids"][0], 0) == 2 for q in Q.index])
    gate = {}
    if 0 < top_ok.mean() < 1:
        gate["auroc"] = float(roc_auc_score(top_ok, top_score))
        print(f"\nRelevance gate: reranker top-1 score predicts 'top passage answers it', AUROC {gate['auroc']:.3f}")
        print("  threshold  retrieve_rate  precision  recall")
        for t in np.percentile(top_score, [10, 25, 40, 50, 60, 75]):
            keep = top_score >= t
            gate[f"{t:.2f}"] = dict(retrieve_rate=float(keep.mean()), precision=float(top_ok[keep].mean()),
                                    recall=float(top_ok[keep].sum() / max(1, top_ok.sum())))
            print(f"  {t:9.2f}  {keep.mean():13.1%}  {top_ok[keep].mean():9.1%}  {top_ok[keep].sum() / max(1, top_ok.sum()):6.1%}")

    M.to_csv(os.path.join(a.out, "methods.csv"))
    Q.drop(columns=["reference"]).to_csv(os.path.join(a.out, "per_query.csv"))
    json.dump(dict(n_queries=len(Q), coverage=float(Q.covered.mean()), best_method=best, gate=gate,
                   methods=M.round(4).to_dict(orient="index")), open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print("\nsaved to", a.out)


# ----------------------------------------------------------------------------- human review
def stage_review(a, queries):
    """Write a CSV to grade by hand: a stratified sample of the judge's grades (blind: the judge's grade is not
    in this file), plus the covered / not-covered split of the queries. Fill the human_grade column with 0 / 1 / 2."""
    import random
    runs = {json.loads(l)["qid"]: json.loads(l) for l in open(os.path.join(a.out, "runs.jsonl"))}
    rel = load_relevance(os.path.join(a.out, "relevance.jsonl"))
    qmap = {q["qid"]: q for q in queries}
    rng = random.Random(42)
    by_grade = {0: [], 1: [], 2: []}
    for qid, g in rel.items():
        for d, v in g.items():
            by_grade[v].append((qid, d))
    per = a.review // 3
    sample = []
    for v in (0, 1, 2):
        sample += rng.sample(by_grade[v], min(per, len(by_grade[v])))
    rng.shuffle(sample)
    text = {}
    for qid, d in sample:
        for m in runs[qid]["methods"].values():
            if d in m["passages"]:
                text[(qid, d)] = m["passages"][d]; break
    path_blind = os.path.join(a.out, "review_blind.csv")
    path_key = os.path.join(a.out, "review_key.csv")
    import csv
    with open(path_blind, "w", newline="", encoding="utf-8") as f, open(path_key, "w", newline="", encoding="utf-8") as k:
        w = csv.writer(f); kw = csv.writer(k)
        w.writerow(["row", "age", "question", "passage", "human_grade"])
        kw.writerow(["row", "qid", "passage_id", "judge_grade"])
        for i, (qid, d) in enumerate(sample, 1):
            w.writerow([i, qmap[qid]["age"], qmap[qid]["question"], text[(qid, d)][:900], ""])
            kw.writerow([i, qid, d, rel[qid][d]])
    # queries nobody answered: top-3 of the best method, to check the corpus really lacks the answer
    miss = [qid for qid, g in rel.items() if 2 not in g.values()]
    picked = rng.sample(miss, min(a.review // 3, len(miss)))
    path_miss = os.path.join(a.out, "review_uncovered.csv")
    with open(path_miss, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "age", "question", "top1", "top2", "top3", "corpus_has_answer (y/n)"])
        for qid in picked:
            ps = list(runs[qid]["methods"]["vector"]["passages"].values())[:3]
            w.writerow([qid, qmap[qid]["age"], qmap[qid]["question"]] + [p[:350] for p in ps] + [""])
    print(f"wrote {path_blind} ({len(sample)} rows to grade), {path_key} (keep closed until done), "
          f"{path_miss} ({len(picked)} of {len(miss)} uncovered queries)")


def stage_agree(a, queries):
    """Compare the filled review_blind.csv with the judge: agreement, Cohen's kappa, confusion matrix, and the
    agreement on the one decision that matters (grade 2 or not)."""
    import csv
    from collections import Counter
    key = {r["row"]: int(r["judge_grade"]) for r in csv.DictReader(open(os.path.join(a.out, "review_key.csv")))}
    hum = {r["row"]: r["human_grade"].strip() for r in csv.DictReader(open(os.path.join(a.out, a.labels)))}
    pairs = [(key[r], int(h)) for r, h in hum.items() if h in ("0", "1", "2")]
    n = len(pairs)
    if not n:
        raise SystemExit("no human_grade filled in review_blind.csv")
    agree = sum(j == h for j, h in pairs) / n
    cj, ch = Counter(j for j, _ in pairs), Counter(h for _, h in pairs)
    pe = sum(cj[g] * ch[g] for g in (0, 1, 2)) / n ** 2
    kappa = (agree - pe) / (1 - pe) if pe < 1 else float("nan")
    bin_agree = sum((j == 2) == (h == 2) for j, h in pairs) / n
    print(f"{n} graded by hand | exact agreement {agree:.1%} | Cohen kappa {kappa:.2f} | "
          f"agreement on 'grade 2 or not' {bin_agree:.1%}")
    print("rows: judge grade, columns: human grade")
    print("      " + "  ".join(f"h={g}" for g in (0, 1, 2)))
    for j in (0, 1, 2):
        print(f"j={j}   " + "  ".join(f"{sum(1 for a_, b_ in pairs if a_ == j and b_ == h):3d}" for h in (0, 1, 2)))
    lenient = sum(j > h for j, h in pairs); strict = sum(j < h for j, h in pairs)
    print(f"judge more generous than human: {lenient}, stricter: {strict}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["retrieve", "judge", "report", "review", "agree", "all"], default="all")
    ap.add_argument("--db", default="rag/lancedb_data")
    ap.add_argument("--table", default="wiki_science_v2_fixed")
    ap.add_argument("--splits", default="data/children_sft_v2_splits")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results/retrieval_v2")
    ap.add_argument("--model", default="gemini-3.5-flash-lite", help="judge model")
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N pending queries")
    ap.add_argument("--router", default=None, help="router.joblib: report only the queries it sends to retrieval")
    ap.add_argument("--router-threshold", dest="router_threshold", type=float, default=0.4)
    ap.add_argument("--labels", default="review_blind.csv", help="filled review file (in --out) for --stage agree")
    ap.add_argument("--review", type=int, default=60, help="rows to grade by hand in --stage review")
    ap.add_argument("--domains", default="results/hallucination_v2/judgments.jsonl",
                    help="hallucination judgments, for each question's domain")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    queries = load_queries(a.splits, a.split)
    if a.stage in ("retrieve", "all"):
        stage_retrieve(a, queries)
    if a.stage in ("judge", "all"):
        stage_judge(a, queries)
    if a.stage in ("report", "all"):
        stage_report(a, queries)
    if a.stage == "review":
        stage_review(a, queries)
    if a.stage == "agree":
        stage_agree(a, queries)


if __name__ == "__main__":
    main()
