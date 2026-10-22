import unittest

from retrieval.fusion import reciprocal_rank_fusion


class RRFTests(unittest.TestCase):
    def test_formula(self):
        fused = reciprocal_rank_fusion({"dense": ["a", "b"], "bm25": ["b", "c"]}, k=60)
        scores = {f.chunk_id: f.score for f in fused}
        self.assertAlmostEqual(scores["a"], 1 / 61)
        self.assertAlmostEqual(scores["b"], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(scores["c"], 1 / 62)
        self.assertEqual(fused[0].chunk_id, "b")

    def test_ranks_recorded_and_deduplicated(self):
        fused = reciprocal_rank_fusion({"dense": ["a", "a", "b"], "bm25": ["b"]}, k=0)
        by_id = {f.chunk_id: f for f in fused}
        self.assertEqual(by_id["a"].ranks, {"dense": 1})
        self.assertEqual(by_id["b"].ranks, {"dense": 2, "bm25": 1})
        self.assertEqual(len(fused), 2)

    def test_empty_and_missing_lists(self):
        self.assertEqual(reciprocal_rank_fusion({}), [])
        fused = reciprocal_rank_fusion({"dense": [], "bm25": ["x", "y"]})
        self.assertEqual([f.chunk_id for f in fused], ["x", "y"])
        fused = reciprocal_rank_fusion({"dense": None, "bm25": ["x"]})
        self.assertEqual([f.chunk_id for f in fused], ["x"])

    def test_deterministic_ties(self):
        a = reciprocal_rank_fusion({"dense": ["p", "q"], "bm25": ["q", "p"]})
        b = reciprocal_rank_fusion({"bm25": ["q", "p"], "dense": ["p", "q"]})
        self.assertEqual([f.chunk_id for f in a], [f.chunk_id for f in b])

    def test_weights(self):
        fused = reciprocal_rank_fusion({"dense": ["a"], "bm25": ["b"]}, k=60, weights={"bm25": 2.0})
        self.assertEqual(fused[0].chunk_id, "b")


if __name__ == "__main__":
    unittest.main()
