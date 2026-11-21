"""Knowledge router: does this message need facts (retrieval) or not (chat, stories, feelings)?

A logistic regression on sentence embeddings, trained on the v2 dataset's needs_knowledge labels by
scripts/train_router.py. About 1 ms per message on CPU after the embedding.
"""
import os

import joblib


class KnowledgeRouter:
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found: run python scripts/train_router.py first")
        bundle = joblib.load(path)
        self.clf, self.features, self.meta = bundle["clf"], bundle["features"], bundle.get("meta", {})
        if self.features == "tfidf":
            self.vec = bundle["vectorizer"]
        else:
            from sentence_transformers import SentenceTransformer
            self.vec = SentenceTransformer(self.features, device="cpu")

    def encode(self, texts):
        if self.features == "tfidf":
            return self.vec.transform(texts)
        return self.vec.encode(texts, normalize_embeddings=True)

    def p_knowledge(self, text: str) -> float:
        return float(self.clf.predict_proba(self.encode([text]))[0, 1])


class AlwaysRetrieve:
    """Fallback when no router is trained: every message goes to retrieval."""
    meta = {"note": "no router; every message is treated as a knowledge question"}

    def p_knowledge(self, text: str) -> float:
        return 1.0
