"""Retrieval over the LanceDB Wikipedia science index: vector, full-text, hybrid (min-max fused), or hybrid with a
cross-encoder reranker. Same methods as rag/retrieval_benchmark.py; the agent uses the one the benchmark picks."""
import re
import time
from dataclasses import dataclass
from typing import List

import pandas as pd

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 100
RERANK_POOL = 50


@dataclass
class Passage:
    id: str
    title: str
    text: str
    score: float


def fts_query(text):
    return " ".join(re.findall(r"[A-Za-z0-9]+", text.replace("’", "'"))) or "a"


def minmax(s):
    return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0.0


class Retriever:
    def __init__(self, db_path, table, method="hybrid_rerank", alpha=0.5):
        import lancedb
        from sentence_transformers import SentenceTransformer
        self.table = lancedb.connect(db_path).open_table(table)
        self.method, self.alpha = method, alpha
        self.embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        self.reranker = None
        if method == "hybrid_rerank":
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANK_MODEL, device="cpu")
            # transformers 5.23 + torch 2.15 on macOS CPU: weights loaded as-is give NaN in float32 linear
            # layers; cloning them fixes it
            for p in self.reranker.model.parameters():
                p.data = p.data.clone()
        self.cols = ["id", "topic", "search_text"]

    def _vector(self, q, k):
        df = self.table.search(self.embedder.encode(q).tolist()).metric("cosine").select(self.cols).limit(k).to_pandas()
        df["score"] = 1.0 - df["_distance"]
        return df

    def _fts(self, q, k):
        df = self.table.search(fts_query(q), query_type="fts").select(self.cols).limit(k).to_pandas()
        df["score"] = df["_score"] if "_score" in df else 0.0
        return df

    def _hybrid(self, q):
        vec, fts = self._vector(q, CANDIDATES), self._fts(q, CANDIDATES)
        m = pd.merge(vec[self.cols + ["score"]], fts[self.cols + ["score"]], on="id", how="outer", suffixes=("_v", "_t"))
        m["topic"] = m.topic_v.fillna(m.topic_t)
        m["search_text"] = m.search_text_v.fillna(m.search_text_t)
        m["score"] = (1 - self.alpha) * minmax(m.score_v.fillna(0.0)) + self.alpha * minmax(m.score_t.fillna(0.0))
        return m.sort_values("score", ascending=False)[self.cols + ["score"]]

    def search(self, q: str, k: int) -> List[Passage]:
        if self.method == "vector":
            df = self._vector(q, k)
        elif self.method == "fts":
            df = self._fts(q, k)
        else:
            df = self._hybrid(q)
            if self.reranker is not None:
                df = df.head(RERANK_POOL).copy()
                df["score"] = self.reranker.predict([(q, t) for t in df.search_text.tolist()])
                df = df.sort_values("score", ascending=False)
        return [Passage(str(r.id), str(r.topic), str(r.search_text), float(r.score)) for r in df.head(k).itertuples()]


class StaticRetriever:
    """For tests and demos: returns fixed passages."""

    def __init__(self, passages: List[Passage]):
        self.passages = passages

    def search(self, q, k):
        return self.passages[:k]
