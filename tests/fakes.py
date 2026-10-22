"""In-memory stand-ins for a Chroma collection, embedding model and reranker."""

import numpy as np


class FakeCollection:
    """Implements the subset of the Chroma Collection API the retrieval layer uses."""

    def __init__(self, records):
        # records: list of (id, text, metadata, embedding)
        self.records = {r[0]: r for r in records}
        self.query_calls = 0

    def count(self):
        return len(self.records)

    def get(self, ids=None, include=None, limit=None, offset=None):
        include = include or ["documents", "metadatas"]
        keys = list(ids) if ids is not None else sorted(self.records)
        keys = [k for k in keys if k in self.records]
        if ids is None:
            offset = offset or 0
            keys = keys[offset: offset + limit] if limit else keys[offset:]
        out = {"ids": keys}
        if "documents" in include:
            out["documents"] = [self.records[k][1] for k in keys]
        if "metadatas" in include:
            out["metadatas"] = [self.records[k][2] for k in keys]
        if "embeddings" in include:
            out["embeddings"] = np.array([self.records[k][3] for k in keys], dtype=np.float32)
        return out

    def query(self, query_embeddings, n_results, include):
        self.query_calls += 1
        q = np.asarray(query_embeddings[0], dtype=np.float32)
        scored = sorted(
            ((float(np.sum((q - np.asarray(r[3])) ** 2)), k) for k, r in self.records.items()),
            key=lambda x: (x[0], x[1]),
        )[:n_results]
        ids = [k for _, k in scored]
        return {
            "ids": [ids],
            "documents": [[self.records[k][1] for k in ids]],
            "metadatas": [[self.records[k][2] for k in ids]],
            "distances": [[d for d, _ in scored]],
        }


class FakeEmbedder:
    """Maps a query to a fixed vector (or raises, to simulate failure)."""

    def __init__(self, vectors=None, default=(0.0, 0.0), fail=False):
        self.vectors = vectors or {}
        self.default = default
        self.fail = fail

    def encode(self, texts):
        if self.fail:
            raise RuntimeError("embedding failed")
        return np.array([self.vectors.get(t, self.default) for t in texts], dtype=np.float32)


class FakeReranker:
    def __init__(self, scores=None, fail_load=False, fail_score=False):
        self.scores = scores or {}
        self.fail_load = fail_load
        self.fail_score = fail_score
        self.calls = 0

    @property
    def available(self):
        return not self.fail_load

    def load(self):
        return not self.fail_load

    def score(self, query, passages):
        self.calls += 1
        if self.fail_score:
            raise RuntimeError("rerank failed")
        return [self.scores.get(p, 0.0) for p in passages]
