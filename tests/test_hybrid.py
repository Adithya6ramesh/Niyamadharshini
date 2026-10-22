import os
import shutil
import tempfile
import unittest

from retrieval import BM25Index, HybridRetriever, RetrievalConfig, RetrievalError, Tokenizer
from tests.fakes import FakeCollection, FakeEmbedder, FakeReranker

RECORDS = [
    # id, text, metadata, embedding
    ("doc.pdf_p1_0", "alpha general introduction", {"source": "doc.pdf", "page": "1", "chunk_id": "0"}, (0.0, 0.1)),
    ("doc.pdf_p2_1", "beta appeal thirty days", {"source": "doc.pdf", "page": "2", "chunk_id": "1"}, (0.0, 0.2)),
    ("doc.pdf_p3_2", "gamma 79/Leg.C3/2025/Law notification", {"source": "doc.pdf", "page": "3", "chunk_id": "2"}, (5.0, 5.0)),
    ("other.pdf_p1_3", "delta unrelated", {"source": "other.pdf", "page": "1", "chunk_id": "0"}, (0.0, 0.3)),
]


class HybridRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.collection = FakeCollection(RECORDS)
        self.index = BM25Index(os.path.join(self.dir, "i.json.gz"), Tokenizer(), collection_name="t")
        self.index.sync(self.collection)
        self.embedder = FakeEmbedder(default=(0.0, 0.0))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def retriever(self, reranker=None, embedder=None, index="default", **cfg):
        config = RetrievalConfig(**{"dense_k": 2, "bm25_k": 2, "final_top_k": 3, "rerank_candidates": 4,
                                    "rerank_enabled": reranker is not None, **cfg})
        return HybridRetriever(config, self.collection, embedder or self.embedder,
                               bm25_index=self.index if index == "default" else index, reranker=reranker)

    def test_dense_mode_matches_original_behaviour(self):
        res = self.retriever(mode="dense").retrieve("anything", top_k=2)
        self.assertEqual([c.chunk_id for c in res.candidates], ["doc.pdf_p1_0", "doc.pdf_p2_1"])
        self.assertAlmostEqual(res.candidates[0].distance, 0.01, places=5)

    def test_hybrid_surfaces_lexical_only_hit_with_metadata_and_distance(self):
        # Dense alone never finds p3_2 (far embedding); BM25 does via the identifier.
        res = self.retriever().retrieve("79/Leg.C3/2025/Law")
        by_id = {c.chunk_id: c for c in res.candidates}
        self.assertIn("doc.pdf_p3_2", by_id)
        hit = by_id["doc.pdf_p3_2"]
        self.assertEqual(hit.metadata["page"], "3")
        self.assertEqual(hit.text, RECORDS[2][1])
        self.assertIsNone(hit.dense_rank)
        self.assertEqual(hit.bm25_rank, 1)
        self.assertAlmostEqual(hit.distance, 50.0, places=4)  # squared L2, like Chroma's "l2" space

    def test_candidates_are_unique(self):
        res = self.retriever().retrieve("beta appeal")
        ids = [c.chunk_id for c in res.candidates]
        self.assertEqual(len(ids), len(set(ids)))

    def test_reranker_reorders_and_preserves_metadata(self):
        rr = FakeReranker(scores={RECORDS[3][1]: 9.0, RECORDS[2][1]: 5.0})
        res = self.retriever(reranker=rr, dense_k=4, bm25_k=4).retrieve("79/Leg.C3/2025/Law delta")
        self.assertTrue(res.reranked)
        self.assertEqual(res.candidates[0].chunk_id, "other.pdf_p1_3")
        self.assertEqual(res.candidates[0].metadata["source"], "other.pdf")
        self.assertIn("rerank", res.timings_ms)

    def test_reranker_load_failure_falls_back_to_hybrid(self):
        res = self.retriever(reranker=FakeReranker(fail_load=True)).retrieve("beta appeal")
        self.assertFalse(res.reranked)
        self.assertIn("reranker unavailable", res.warnings)
        self.assertTrue(res.candidates)

    def test_reranker_runtime_failure_falls_back_to_hybrid(self):
        res = self.retriever(reranker=FakeReranker(fail_score=True)).retrieve("beta appeal")
        self.assertFalse(res.reranked)
        self.assertIn("reranking failed", res.warnings)
        self.assertTrue(res.candidates)

    def test_rerank_can_be_disabled_per_call(self):
        rr = FakeReranker()
        res = self.retriever(reranker=rr).retrieve("beta", rerank=False)
        self.assertFalse(res.reranked)
        self.assertEqual(rr.calls, 0)

    def test_missing_bm25_index_falls_back_to_dense(self):
        res = self.retriever(index=None).retrieve("beta appeal")
        self.assertTrue(res.candidates)
        self.assertIn("bm25 index unavailable", res.warnings)

    def test_dense_failure_falls_back_to_bm25(self):
        res = self.retriever(embedder=FakeEmbedder(fail=True)).retrieve("beta appeal")
        self.assertEqual(res.candidates[0].chunk_id, "doc.pdf_p2_1")
        self.assertIn("dense retrieval failed", res.warnings)

    def test_both_failing_raises(self):
        with self.assertRaises(RetrievalError):
            self.retriever(embedder=FakeEmbedder(fail=True), index=None).retrieve("beta")

    def test_empty_and_whitespace_queries(self):
        r = self.retriever()
        self.assertEqual(r.retrieve("").candidates, [])
        self.assertEqual(r.retrieve("   \n\t").candidates, [])
        self.assertEqual(self.collection.query_calls, 0)

    def test_non_string_query_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever().retrieve(None)

    def test_stale_bm25_hit_is_dropped(self):
        self.collection.records.pop("doc.pdf_p3_2")  # deleted from Chroma, index not yet synced
        res = self.retriever().retrieve("79/Leg.C3/2025/Law")
        self.assertNotIn("doc.pdf_p3_2", [c.chunk_id for c in res.candidates])

    def test_bm25_only_mode_still_reports_distance(self):
        res = self.retriever(mode="bm25").retrieve("beta appeal")
        self.assertEqual(res.candidates[0].chunk_id, "doc.pdf_p2_1")
        self.assertIsNotNone(res.candidates[0].distance)


class ConfigTests(unittest.TestCase):
    def test_env_overrides(self):
        env = {"RETRIEVAL_MODE": "dense", "RRF_K": "30", "RERANK_ENABLED": "false", "FINAL_TOP_K": "5"}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            cfg = RetrievalConfig.from_env()
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertEqual((cfg.mode, cfg.rrf_k, cfg.rerank_enabled, cfg.final_top_k), ("dense", 30, False, 5))

    def test_rerank_quantize_default_and_off(self):
        old = os.environ.pop("RERANK_QUANTIZE", None)
        try:
            self.assertEqual(RetrievalConfig.from_env().rerank_quantize, "int8")
            os.environ["RERANK_QUANTIZE"] = "none"
            self.assertEqual(RetrievalConfig.from_env().rerank_quantize, "")
        finally:
            os.environ.pop("RERANK_QUANTIZE", None)
            if old is not None:
                os.environ["RERANK_QUANTIZE"] = old

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            RetrievalConfig(mode="fancy")


if __name__ == "__main__":
    unittest.main()
