"""Add an ANN vector index (IVF-PQ) to the paragraph table and measure what it costs and gains.

Why: wiki_v4_para has no vector index, so every query scans all paragraph vectors (several hundred MB). That is
fast while the vectors sit in memory (0.86 s p50 for the whole retrieval in the benchmark) and slow once the
model servers push them out (4-5 s with the generator and the guard running). IVF-PQ searches only the nearest
partitions over compressed vectors; `refine_factor` re-scores the best candidates with the full vectors.

The index is added to the existing table (a new table version; the data is not rewritten). To go back:
`python -c "import lancedb; t=lancedb.connect('rag/lancedb_data').open_table('wiki_v4_para'); t.drop_index('vector_idx')"`

    python rag/build_ann_index.py                 # sprout-index env; -> results/retrieval_v4/ann_index.json
    python rag/build_ann_index.py --skip-build    # measure again without rebuilding

Reported per setting (nprobes, refine_factor): recall@20 and @100 against exact search (the agent reranks the
top 20 of the fused list, so recall@20 of the vector leg is what matters), and p50 / p95 query latency.
Measure with the model servers running too, since that is where the brute-force scan was slow.
"""
import argparse, csv, json, math, os, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="rag/lancedb_data")
ap.add_argument("--table", default="wiki_v4_para")
ap.add_argument("--queries", default="results/retrieval_v3/per_query.csv")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--skip-build", action="store_true")
ap.add_argument("--nprobes", default="10,20,50")
ap.add_argument("--refine", default="0,10", help="0 = no refine")
ap.add_argument("--no-exact", action="store_true", help="skip the exact scan (slow under load); latency only, no recall")
ap.add_argument("--out", default="results/retrieval_v4/ann_index.json")
a = ap.parse_args()

import lancedb
from sentence_transformers import SentenceTransformer
t = lancedb.connect(a.db).open_table(a.table)
n_rows = t.count_rows()
print(f"{a.table}: {n_rows} rows; indexes: {t.list_indices()}")
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
for p in m.parameters():
    p.data = p.data.clone()                              # macOS misaligned-weights workaround
qs = [r["question"] for r in csv.DictReader(open(a.queries, encoding="utf-8"))][: a.n]
V = m.encode(qs, convert_to_numpy=True)


def run(fn, k=100):
    lat, ids = [], []
    for v in V:
        t0 = time.perf_counter(); df = fn(v.tolist()).limit(k).select(["id"]).to_pandas()
        lat.append(time.perf_counter() - t0); ids.append(df.id.tolist())
    return ids, lat


exact = lambda v: t.search(v).metric("cosine").bypass_vector_index()
res = dict(table=a.table, rows=n_rows, queries=len(qs))
exact_ids = None
if not a.no_exact:
    run(exact, 5)                                        # warm up
    exact_ids, exact_lat = run(exact)
    res["exact"] = dict(p50_ms=1000 * np.median(exact_lat), p95_ms=1000 * np.percentile(exact_lat, 95))
    print(f"exact scan: p50 {res['exact']['p50_ms']:.0f} ms, p95 {res['exact']['p95_ms']:.0f} ms")

parts = int(2 ** round(math.log2(max(16, math.sqrt(n_rows)))))   # ~sqrt(rows), power of two
if not a.skip_build:
    t0 = time.perf_counter()
    try:
        t.create_index(metric="cosine", vector_column_name="vector", num_partitions=parts, num_sub_vectors=48,
                       replace=True)
    except TypeError:                                    # newer lancedb API
        t.create_index(metric="cosine", vector_column_name="vector", index_type="IVF_PQ",
                       num_partitions=parts, num_sub_vectors=48, replace=True)
    res["build_seconds"] = time.perf_counter() - t0
    print(f"built IVF-PQ ({parts} partitions, 48 sub-vectors) in {res['build_seconds']:.0f} s")
res.update(num_partitions=parts, num_sub_vectors=48, settings=[])

recall = lambda got, k: (np.mean([len(set(g[:k]) & set(e[:k])) / k for g, e in zip(got, exact_ids)])
                         if exact_ids else float("nan"))
for nprobes in map(int, a.nprobes.split(",")):
    for refine in [int(x) or None for x in a.refine.split(",")]:
        def ann(v, nprobes=nprobes, refine=refine):
            q = t.search(v).metric("cosine").nprobes(nprobes)
            return q.refine_factor(refine) if refine else q
        run(ann, 5)
        ids, lat = run(ann)
        s = dict(nprobes=nprobes, refine_factor=refine, recall_at_20=recall(ids, 20), recall_at_100=recall(ids, 100),
                 p50_ms=1000 * np.median(lat), p95_ms=1000 * np.percentile(lat, 95))
        res["settings"].append(s)
        print(f"nprobes {nprobes:3d} refine {str(refine):4s}: recall@20 {s['recall_at_20']:.3f} "
              f"recall@100 {s['recall_at_100']:.3f}  p50 {s['p50_ms']:.0f} ms  p95 {s['p95_ms']:.0f} ms")

fts = lambda: [t.search(" ".join(q.split()[:12]), query_type="fts").limit(100).select(["id"]).to_pandas() for q in qs[:20]]
t0 = time.perf_counter(); fts(); res["fts_ms_per_query"] = 1000 * (time.perf_counter() - t0) / 20
print(f"full-text search: {res['fts_ms_per_query']:.0f} ms per query (for comparison)")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1, default=float)
print("saved", a.out)
