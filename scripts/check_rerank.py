"""Smoke test: run the benchmark retriever on a few queries and check the reranker scores are finite."""
import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))
from retrieval_benchmark import Retriever

r = Retriever("rag/lancedb_data", "wiki_science_v2_fixed")
for q in ["Why do some animals go extinct?", "Why is the sky blue?", "How do magnets work?"]:
    df, secs = r.run(q)["hybrid_rerank"]
    ok = all(math.isfinite(x) for x in df.score)
    print(f"{'OK ' if ok else 'NaN'} {secs:.2f}s | {q}")
    for t, s in list(zip(df.topic, df.score))[:3]:
        print(f"     {s:7.3f}  {t}")
