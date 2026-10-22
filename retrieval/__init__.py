"""
Retrieval layer for Niyamadharshini: dense (existing) + BM25 + RRF + reranking.

Typical use:

    from retrieval import RetrievalConfig, build_retriever
    retriever = build_retriever(RetrievalConfig.from_env(), embed_model, collection)
    result = retriever.retrieve("question")
"""

import logging

from .bm25_index import BM25Index
from .config import RetrievalConfig
from .hybrid import Candidate, HybridRetriever, RetrievalError, RetrievalResult
from .reranker import CrossEncoderReranker
from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)

__all__ = [
    "BM25Index", "Candidate", "CrossEncoderReranker", "HybridRetriever", "RetrievalConfig",
    "RetrievalError", "RetrievalResult", "Tokenizer", "build_bm25_index", "build_retriever",
]


def build_bm25_index(config: RetrievalConfig, collection, sync: bool = True) -> BM25Index:
    index = BM25Index(
        config.bm25_index_path, Tokenizer(), k1=config.bm25_k1, b=config.bm25_b,
        collection_name=config.collection_name,
    )
    if sync:
        index.sync(collection)
    elif not index.load():
        index.rebuild(collection)
    return index


def build_retriever(config: RetrievalConfig, embed_model, collection) -> HybridRetriever:
    """Wire up the retriever. BM25 and reranker failures degrade instead of crashing startup."""
    bm25 = None
    if config.mode in ("hybrid", "bm25"):
        try:
            bm25 = build_bm25_index(config, collection, sync=config.bm25_auto_sync)
        except Exception:
            logger.exception("BM25 index could not be prepared; continuing without lexical retrieval")

    reranker = None
    if config.rerank_enabled:
        reranker = CrossEncoderReranker(
            config.reranker_model, max_length=config.rerank_max_length,
            batch_size=config.rerank_batch_size, device=config.reranker_device,
            quantize=config.rerank_quantize,
        )
        reranker.load()

    logger.info("Retriever ready: mode=%s dense_k=%d bm25_k=%d rrf_k=%d rerank=%s(%s) candidates=%d final_top_k=%d",
                config.mode, config.dense_k, config.bm25_k, config.rrf_k, config.rerank_enabled,
                config.reranker_model if config.rerank_enabled else "-",
                config.rerank_candidates, config.final_top_k)
    return HybridRetriever(config, collection, embed_model, bm25_index=bm25, reranker=reranker)
