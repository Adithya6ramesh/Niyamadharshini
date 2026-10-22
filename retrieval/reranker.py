"""
Cross-encoder reranker (default: BAAI/bge-reranker-v2-m3).

Chosen because it is Apache-2.0, multilingual, built on the same XLM-R / bge-m3
backbone as the embedding model (XLM-R pretraining covers Malayalam), and loads
through sentence-transformers' CrossEncoder, which the project already uses.
Any CrossEncoder-compatible model can be swapped in with RERANKER_MODEL.

The model is loaded once. If loading fails, `available` stays False and the
retriever falls back to the fused hybrid ranking.
"""

import logging
import threading
import time
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 512, batch_size: int = 8, device: str = "",
                 quantize: str = ""):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or None
        self.quantize = quantize  # "" or "int8" (dynamic int8 Linear layers; CPU only, ~1.6x faster)
        self._model = None
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def load(self) -> bool:
        with self._lock:
            if self._model is not None:
                return True
            if self._load_error is not None:
                return False  # do not retry a failed load on every request
            start = time.perf_counter()
            try:
                from sentence_transformers import CrossEncoder
                model = CrossEncoder(self.model_name, max_length=self.max_length, device=self.device)
                if self.quantize == "int8" and model.device.type != "cpu":
                    logger.warning("int8 quantization is CPU-only; running the reranker at full precision on %s",
                                   model.device)
                elif self.quantize == "int8":
                    import warnings
                    import torch
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        model.model = torch.ao.quantization.quantize_dynamic(
                            model.model, {torch.nn.Linear}, dtype=torch.qint8)
                elif self.quantize:
                    raise ValueError(f"Unsupported RERANK_QUANTIZE value {self.quantize!r}")
                self._model = model
                logger.info("Reranker %s loaded on %s%s in %.1fs", self.model_name, model.device,
                            " (int8)" if self.quantize else "", time.perf_counter() - start)
                return True
            except Exception as e:  # missing weights, no network, OOM, incompatible model ...
                self._load_error = f"{type(e).__name__}: {e}"
                logger.error("Reranker %s failed to load; falling back to hybrid ranking. %s",
                             self.model_name, self._load_error)
                return False

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        if self._model is None:
            raise RuntimeError("Reranker is not loaded")
        if not passages:
            return []
        scores = self._model.predict(
            [(query, p) for p in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]
