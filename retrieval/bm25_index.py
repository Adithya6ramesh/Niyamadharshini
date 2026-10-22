"""
Persistent BM25 index over the chunks already stored in ChromaDB.

* Chroma remains the single source of truth: the index stores only per-chunk
  term frequencies and a content hash keyed by the Chroma record ID, never the
  chunk text or metadata. Text/metadata for BM25 hits are read back from Chroma.
* `sync()` diffs the index against the collection (added / removed / changed
  chunks) and applies only the differences, so it is cheap to run at startup
  and after ingestion. If the index file is lost or was built with a different
  tokenizer, it is rebuilt from the collection.
* Writes are atomic (temp file + os.replace), so a crash cannot leave a
  half-written index behind.
"""

import gzip
import hashlib
import json
import logging
import math
import os
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)

INDEX_FORMAT = 1
_FETCH_BATCH = 500


def content_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


@dataclass
class SyncStats:
    added: int = 0
    removed: int = 0
    updated: int = 0
    total: int = 0
    rebuilt: bool = False
    seconds: float = 0.0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed or self.updated or self.rebuilt)


class _Postings:
    """Immutable, query-time view of the index (swapped atomically after changes)."""

    def __init__(self, doc_ids: List[str], term_freqs: List[Dict[str, int]], k1: float, b: float):
        self.doc_ids = doc_ids
        n = len(doc_ids)
        doc_len = np.array([sum(tf.values()) for tf in term_freqs], dtype=np.float32)
        avgdl = float(doc_len.mean()) if n and doc_len.sum() > 0 else 1.0

        postings: Dict[str, List[Tuple[int, int]]] = {}
        for i, tf in enumerate(term_freqs):
            for term, count in tf.items():
                postings.setdefault(term, []).append((i, count))

        norm = k1 * (1.0 - b + b * doc_len / avgdl)
        self.terms: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for term, plist in postings.items():
            idx = np.fromiter((p[0] for p in plist), dtype=np.int32, count=len(plist))
            tf = np.fromiter((p[1] for p in plist), dtype=np.float32, count=len(plist))
            df = len(plist)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))  # Lucene-style, always > 0
            self.terms[term] = (idx, (idf * tf * (k1 + 1.0) / (tf + norm[idx])).astype(np.float32))

    def search(self, query_terms: List[str], k: int) -> List[Tuple[str, float]]:
        n = len(self.doc_ids)
        if n == 0 or not query_terms:
            return []
        scores = np.zeros(n, dtype=np.float32)
        matched = False
        for term in set(query_terms):
            entry = self.terms.get(term)
            if entry is None:
                continue
            idx, contrib = entry
            scores[idx] += contrib
            matched = True
        if not matched:
            return []
        k = min(k, n)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[scores[top] > 0]
        # Sort by score desc, then by chunk ID for deterministic ties
        ordered = sorted(top.tolist(), key=lambda i: (-float(scores[i]), self.doc_ids[i]))
        return [(self.doc_ids[i], float(scores[i])) for i in ordered]


class BM25Index:
    def __init__(self, path: str, tokenizer: Optional[Tokenizer] = None,
                 k1: float = 1.2, b: float = 0.75, collection_name: str = ""):
        self.path = path
        self.tokenizer = tokenizer or Tokenizer()
        self.k1 = k1
        self.b = b
        self.collection_name = collection_name
        self._docs: Dict[str, Dict] = {}  # chunk_id -> {"h": hash, "tf": {term: count}}
        self._view = _Postings([], [], k1, b)
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._view.doc_ids)

    # ---------- query ----------

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Return up to k (chunk_id, bm25_score) pairs, best first."""
        view = self._view  # snapshot; safe against concurrent sync
        return view.search(self.tokenizer.tokenize(query), k)

    # ---------- maintenance ----------

    def add_or_update(self, chunk_id: str, text: str) -> None:
        with self._lock:
            self._docs[chunk_id] = {"h": content_hash(text), "tf": dict(Counter(self.tokenizer.tokenize(text)))}

    def remove(self, chunk_id: str) -> None:
        with self._lock:
            self._docs.pop(chunk_id, None)

    def refresh(self) -> None:
        """Rebuild the query-time view after add/update/remove calls."""
        with self._lock:
            ids = sorted(self._docs)
            self._view = _Postings(ids, [self._docs[i]["tf"] for i in ids], self.k1, self.b)

    def sync(self, collection, save: bool = True) -> SyncStats:
        """Bring the index in line with the Chroma collection (incremental)."""
        start = time.perf_counter()
        stats = SyncStats()
        with self._lock:
            loaded = self._docs or self.load()
            stats.rebuilt = not loaded

            current: Dict[str, str] = {}
            for ids, documents in _iter_collection(collection):
                for chunk_id, text in zip(ids, documents):
                    h = content_hash(text)
                    current[chunk_id] = h
                    existing = self._docs.get(chunk_id)
                    if existing is None:
                        stats.added += 1
                        self.add_or_update(chunk_id, text)
                    elif existing["h"] != h:
                        stats.updated += 1
                        self.add_or_update(chunk_id, text)

            for chunk_id in [c for c in self._docs if c not in current]:
                self.remove(chunk_id)
                stats.removed += 1

            stats.total = len(self._docs)
            if stats.changed or len(self._view.doc_ids) != stats.total:
                self.refresh()
            if save and stats.changed:
                self.save()
        stats.seconds = time.perf_counter() - start
        logger.info("BM25 sync: total=%d added=%d updated=%d removed=%d rebuilt=%s (%.2fs)",
                    stats.total, stats.added, stats.updated, stats.removed, stats.rebuilt, stats.seconds)
        return stats

    def rebuild(self, collection, save: bool = True) -> SyncStats:
        with self._lock:
            self._docs = {}
            self._view = _Postings([], [], self.k1, self.b)
            stats = SyncStats(rebuilt=True)
            start = time.perf_counter()
            for ids, documents in _iter_collection(collection):
                for chunk_id, text in zip(ids, documents):
                    self.add_or_update(chunk_id, text)
            stats.added = stats.total = len(self._docs)
            self.refresh()
            if save:
                self.save()
            stats.seconds = time.perf_counter() - start
        return stats

    # ---------- persistence ----------

    def _header(self) -> Dict:
        return {
            "format": INDEX_FORMAT,
            "tokenizer": self.tokenizer.version,
            "collection": self.collection_name,
        }

    def save(self) -> None:
        with self._lock:
            payload = {**self._header(), "docs": self._docs}
            directory = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".bm25-", suffix=".tmp", dir=directory)
            try:
                with os.fdopen(fd, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb") as gz:
                    gz.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
                os.replace(tmp, self.path)
            except BaseException:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise

    def load(self) -> bool:
        """Load from disk. Returns False (index left empty) if missing, stale or unreadable."""
        if not os.path.exists(self.path):
            logger.info("BM25 index not found at %s; it will be built from ChromaDB", self.path)
            return False
        try:
            with gzip.open(self.path, "rb") as f:
                payload = json.loads(f.read().decode("utf-8"))
        except (OSError, ValueError) as e:
            logger.warning("BM25 index at %s is unreadable (%s); rebuilding", self.path, e)
            return False
        header = self._header()
        for key in ("format", "tokenizer", "collection"):
            if payload.get(key) != header[key]:
                logger.info("BM25 index %s changed (%r -> %r); rebuilding", key, payload.get(key), header[key])
                return False
        with self._lock:
            self._docs = payload.get("docs", {})
            self.refresh()
        return True


def _iter_collection(collection):
    """Yield (ids, documents) pages from a Chroma collection."""
    offset = 0
    while True:
        page = collection.get(include=["documents"], limit=_FETCH_BATCH, offset=offset)
        ids = page.get("ids") or []
        if not ids:
            break
        yield ids, [d or "" for d in (page.get("documents") or [""] * len(ids))]
        if len(ids) < _FETCH_BATCH:
            break
        offset += len(ids)
