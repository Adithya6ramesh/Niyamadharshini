"""
Hybrid retrieval: existing dense search + BM25 -> RRF -> cross-encoder rerank.

The dense path is the original implementation, unchanged: BGE-M3 query
embedding + `collection.query` on the existing Chroma collection.

Degradation is explicit and logged:
  * BM25 unavailable or failing     -> dense ranking only
  * dense (embedding/Chroma) failing -> BM25 ranking only
  * reranker unavailable or failing -> fused RRF ranking
  * both retrievers failing         -> RetrievalError
Only chunk IDs, counts and timings are logged, never document or query text.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import RetrievalConfig
from .fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    pass


@dataclass
class Candidate:
    chunk_id: str
    text: str
    metadata: Dict
    distance: Optional[float] = None       # Chroma l2 distance to the query (same as before)
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


@dataclass
class RetrievalResult:
    candidates: List[Candidate]
    mode: str
    reranked: bool = False
    timings_ms: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class HybridRetriever:
    def __init__(self, config: RetrievalConfig, collection, embed_model, bm25_index=None, reranker=None):
        self.config = config
        self.collection = collection
        self.embed_model = embed_model
        self.bm25_index = bm25_index
        self.reranker = reranker

    def retrieve(self, query: str, top_k: Optional[int] = None, mode: Optional[str] = None,
                 rerank: Optional[bool] = None) -> RetrievalResult:
        cfg = self.config
        mode = mode or cfg.mode
        top_k = top_k or cfg.final_top_k
        use_rerank = cfg.rerank_enabled if rerank is None else rerank
        result = RetrievalResult(candidates=[], mode=mode)
        t = result.timings_ms
        t_start = time.perf_counter()

        if not isinstance(query, str):
            raise ValueError("query must be a string")
        query = " ".join(query.split())[: cfg.max_query_chars]
        if not query:
            return result

        records: Dict[str, Candidate] = {}
        ranked: Dict[str, List[str]] = {}
        query_embedding = None
        dense_failed = bm25_failed = False

        # ---- dense (existing implementation) ----
        if mode in ("hybrid", "dense"):
            try:
                s = time.perf_counter()
                query_embedding = self._embed(query)
                t["embed"] = _ms(s)
                s = time.perf_counter()
                res = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=cfg.dense_k if mode == "hybrid" else max(cfg.dense_k, top_k),
                    include=["documents", "metadatas", "distances"],
                )
                t["dense"] = _ms(s)
                ids = res["ids"][0]
                for rank, (cid, doc, meta, dist) in enumerate(
                        zip(ids, res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1):
                    records[cid] = Candidate(cid, doc or "", meta or {}, float(dist), dense_rank=rank)
                ranked["dense"] = list(ids)
            except Exception as e:
                dense_failed = True
                result.warnings.append("dense retrieval failed")
                logger.exception("Dense retrieval failed: %s", type(e).__name__)

        # ---- BM25 ----
        if mode in ("hybrid", "bm25"):
            if self.bm25_index is None or len(self.bm25_index) == 0:
                bm25_failed = True
                result.warnings.append("bm25 index unavailable")
                logger.warning("BM25 index unavailable; skipping lexical retrieval")
            else:
                try:
                    s = time.perf_counter()
                    k = cfg.bm25_k if mode == "hybrid" else max(cfg.bm25_k, top_k)
                    hits = self.bm25_index.search(query, k)
                    t["bm25"] = _ms(s)
                    ranked["bm25"] = [cid for cid, _ in hits]
                    for rank, (cid, score) in enumerate(hits, start=1):
                        cand = records.get(cid)
                        if cand is None:
                            cand = records[cid] = Candidate(cid, "", {})
                        cand.bm25_rank, cand.bm25_score = rank, score
                except Exception as e:
                    bm25_failed = True
                    result.warnings.append("bm25 retrieval failed")
                    logger.exception("BM25 retrieval failed: %s", type(e).__name__)

        if mode == "hybrid" and dense_failed and bm25_failed:
            raise RetrievalError("Both dense and BM25 retrieval failed")
        if mode == "dense" and dense_failed:
            raise RetrievalError("Dense retrieval failed")
        if mode == "bm25" and bm25_failed:
            raise RetrievalError("BM25 retrieval failed")

        # ---- fusion ----
        s = time.perf_counter()
        fused = reciprocal_rank_fusion(ranked, k=cfg.rrf_k)
        t["fusion"] = _ms(s)
        pool_size = max(top_k, cfg.rerank_candidates) if use_rerank else top_k
        pool: List[Candidate] = []
        for item in fused[:pool_size]:
            cand = records[item.chunk_id]
            cand.rrf_score = item.score
            pool.append(cand)

        # ---- hydrate BM25-only hits from Chroma (text, metadata, distance) ----
        missing = [c for c in pool if c.dense_rank is None]
        if missing:
            s = time.perf_counter()
            pool = self._hydrate(pool, missing, query, query_embedding, result)
            t["hydrate"] = _ms(s)

        # ---- rerank ----
        if use_rerank and pool:
            if self.reranker is not None and self.reranker.load():
                try:
                    s = time.perf_counter()
                    scores = self.reranker.score(query, [c.text for c in pool])
                    t["rerank"] = _ms(s)
                    for cand, score in zip(pool, scores):
                        cand.rerank_score = score
                    # stable sort keeps fused order for equal scores
                    pool = sorted(pool, key=lambda c: -c.rerank_score)
                    result.reranked = True
                except Exception as e:
                    result.warnings.append("reranking failed")
                    logger.exception("Reranking failed (%s); using hybrid ranking", type(e).__name__)
            else:
                result.warnings.append("reranker unavailable")

        result.candidates = pool[:top_k]
        t["total"] = _ms(t_start)
        logger.info(
            "retrieve mode=%s reranked=%s query_chars=%d dense=%d bm25=%d fused=%d returned=%d timings_ms=%s ids=%s",
            mode, result.reranked, len(query), len(ranked.get("dense", [])), len(ranked.get("bm25", [])),
            len(fused), len(result.candidates), {k: round(v, 1) for k, v in t.items()},
            [c.chunk_id for c in result.candidates],
        )
        return result

    # ------------------------------------------------------------------

    def _embed(self, query: str) -> List[float]:
        return self.embed_model.encode([query])[0].tolist()

    def _hydrate(self, pool, missing, query, query_embedding, result) -> List[Candidate]:
        include = ["documents", "metadatas", "embeddings"]
        try:
            got = self.collection.get(ids=[c.chunk_id for c in missing], include=include)
        except Exception as e:
            logger.exception("Failed to load BM25 hits from ChromaDB: %s", type(e).__name__)
            result.warnings.append("failed to load lexical hits")
            return [c for c in pool if c.dense_rank is not None]

        by_id = {}
        embeddings = got.get("embeddings")
        for i, cid in enumerate(got.get("ids") or []):
            emb = embeddings[i] if embeddings is not None and len(embeddings) > i else None
            by_id[cid] = (got["documents"][i], got["metadatas"][i], emb)

        if query_embedding is None:
            try:
                query_embedding = self._embed(query)
            except Exception:
                logger.warning("Could not embed query to compute distances for lexical hits")

        q = np.asarray(query_embedding, dtype=np.float32) if query_embedding is not None else None
        kept = []
        for cand in pool:
            if cand.dense_rank is not None:
                kept.append(cand)
                continue
            rec = by_id.get(cand.chunk_id)
            if rec is None:
                # Index is stale (chunk removed from Chroma); drop it rather than cite a ghost
                logger.warning("BM25 hit %s not found in ChromaDB; index may need sync", cand.chunk_id)
                continue
            doc, meta, emb = rec
            cand.text, cand.metadata = doc or "", meta or {}
            if q is not None and emb is not None:
                # Chroma's "l2" space reports squared Euclidean distance
                diff = q - np.asarray(emb, dtype=np.float32)
                cand.distance = float(np.dot(diff, diff))
            kept.append(cand)
        return kept


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
