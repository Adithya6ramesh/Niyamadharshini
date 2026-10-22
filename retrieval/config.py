"""
Centralized retrieval configuration.

Every value can be overridden with an environment variable (see .env.example),
so retrieval modes can be switched for debugging and evaluation without code
changes.
"""

import os
from dataclasses import dataclass, replace


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}")


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"Environment variable {name} must be a number, got {value!r}")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _quantize_setting(value: str) -> str:
    value = value.strip().lower()
    return "" if value in ("none", "off", "false", "fp32", "0") else value


RETRIEVAL_MODES = ("hybrid", "dense", "bm25")


@dataclass(frozen=True)
class RetrievalConfig:
    # Existing vector store (read-only from the retrieval layer's point of view)
    chroma_path: str = "chroma_db"
    collection_name: str = "malayalam_docs"
    embed_model_name: str = "BAAI/bge-m3"

    # "hybrid" = dense + BM25 fused with RRF; "dense" / "bm25" = single retriever
    mode: str = "hybrid"

    # Candidate depth per retriever and RRF smoothing constant
    dense_k: int = 30
    bm25_k: int = 30
    rrf_k: int = 60

    # Number of chunks handed to the LLM
    final_top_k: int = 10

    # BM25 lexical index
    bm25_index_path: str = "bm25_index/malayalam_docs.json.gz"
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    bm25_auto_sync: bool = True

    # Cross-encoder reranking
    rerank_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_candidates: int = 30
    rerank_max_length: int = 512
    rerank_batch_size: int = 8
    reranker_device: str = ""  # empty = auto (cuda if available, else cpu)
    # "int8" = dynamic int8 on CPU (default). On the golden set it was 2.4x faster than full
    # precision (10.5s vs 24.9s per query) with the same rank-1 hits (13/17) and one fewer
    # top-10 hit (16/17 vs 17/17). "" = full precision (RERANK_QUANTIZE=none); GPUs skip int8.
    rerank_quantize: str = "int8"

    # Defensive cap on query length (characters)
    max_query_chars: int = 2000

    def __post_init__(self):
        if self.mode not in RETRIEVAL_MODES:
            raise ValueError(f"RETRIEVAL_MODE must be one of {RETRIEVAL_MODES}, got {self.mode!r}")
        for name in ("dense_k", "bm25_k", "final_top_k", "rerank_candidates", "rerank_batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.rrf_k < 0:
            raise ValueError("rrf_k must be >= 0")

    @classmethod
    def from_env(cls, **overrides) -> "RetrievalConfig":
        d = cls.__dataclass_fields__
        config = cls(
            chroma_path=_env_str("CHROMA_DB_PATH", d["chroma_path"].default),
            collection_name=_env_str("CHROMA_COLLECTION", d["collection_name"].default),
            embed_model_name=_env_str("EMBED_MODEL_NAME", d["embed_model_name"].default),
            mode=_env_str("RETRIEVAL_MODE", d["mode"].default).lower(),
            dense_k=_env_int("DENSE_TOP_K", d["dense_k"].default),
            bm25_k=_env_int("BM25_TOP_K", d["bm25_k"].default),
            rrf_k=_env_int("RRF_K", d["rrf_k"].default),
            final_top_k=_env_int("FINAL_TOP_K", d["final_top_k"].default),
            bm25_index_path=_env_str("BM25_INDEX_PATH", d["bm25_index_path"].default),
            bm25_k1=_env_float("BM25_K1", d["bm25_k1"].default),
            bm25_b=_env_float("BM25_B", d["bm25_b"].default),
            bm25_auto_sync=_env_bool("BM25_AUTO_SYNC", d["bm25_auto_sync"].default),
            rerank_enabled=_env_bool("RERANK_ENABLED", d["rerank_enabled"].default),
            reranker_model=_env_str("RERANKER_MODEL", d["reranker_model"].default),
            rerank_candidates=_env_int("RERANK_CANDIDATES", d["rerank_candidates"].default),
            rerank_max_length=_env_int("RERANK_MAX_LENGTH", d["rerank_max_length"].default),
            rerank_batch_size=_env_int("RERANK_BATCH_SIZE", d["rerank_batch_size"].default),
            reranker_device=_env_str("RERANKER_DEVICE", d["reranker_device"].default),
            rerank_quantize=_quantize_setting(_env_str("RERANK_QUANTIZE", d["rerank_quantize"].default)),
        )
        return replace(config, **overrides) if overrides else config
