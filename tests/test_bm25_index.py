import gzip
import json
import os
import shutil
import tempfile
import unittest

from retrieval.bm25_index import BM25Index
from retrieval.tokenizer import Tokenizer
from tests.fakes import FakeCollection


def make_collection(docs):
    return FakeCollection([(cid, text, {"source": "x.pdf", "page": "1"}, (0.0, 0.0)) for cid, text in docs])


class BM25IndexTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "idx.json.gz")
        self.docs = [
            ("a", "The defaulter may request the Collector to limit the attachment"),
            ("b", "An appeal shall be disposed of within thirty days"),
            ("c", "സര്‍ക്കാര്‍ ഉത്തരവ്‌ നമ്പര്‍ 18/ലെഗ്‌.സി3/2024/നിയമം"),
        ]
        self.collection = make_collection(self.docs)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def new_index(self, tokenizer=None):
        return BM25Index(self.path, tokenizer or Tokenizer(), collection_name="test")

    def test_build_and_search(self):
        idx = self.new_index()
        stats = idx.sync(self.collection)
        self.assertTrue(stats.rebuilt)
        self.assertEqual(stats.total, 3)
        self.assertEqual(idx.search("collector attachment", 3)[0][0], "a")
        self.assertEqual(idx.search("thirty days appeal", 3)[0][0], "b")

    def test_modern_keyboard_query_matches_legacy_corpus(self):
        idx = self.new_index()
        idx.sync(self.collection)
        self.assertEqual(idx.search("സർക്കാർ ഉത്തരവ്", 3)[0][0], "c")
        self.assertEqual(idx.search("18/ലെഗ്.സി3/2024/നിയമം", 3)[0][0], "c")

    def test_no_match_returns_empty(self):
        idx = self.new_index()
        idx.sync(self.collection)
        self.assertEqual(idx.search("electric vehicle subsidy", 5), [])
        self.assertEqual(idx.search("", 5), [])

    def test_persistence_and_incremental_sync(self):
        idx = self.new_index()
        idx.sync(self.collection)
        self.assertTrue(os.path.exists(self.path))

        # restart: loads from disk, nothing to do
        idx2 = self.new_index()
        stats = idx2.sync(self.collection)
        self.assertFalse(stats.rebuilt)
        self.assertFalse(stats.changed)
        self.assertEqual(len(idx2), 3)

        # add, update and remove chunks in the collection
        changed = make_collection([
            ("a", "The defaulter may request the Collector to limit the attachment"),
            ("b", "A revision petition within sixty days"),
            ("d", "Horti-wine inserted in the schedule"),
        ])
        stats = idx2.sync(changed)
        self.assertEqual((stats.added, stats.updated, stats.removed), (1, 1, 1))
        self.assertEqual(idx2.search("sixty days revision", 1)[0][0], "b")
        self.assertEqual(idx2.search("horti-wine", 1)[0][0], "d")
        self.assertEqual(idx2.search("ഉത്തരവ്", 5), [])

    def test_index_stores_no_raw_text(self):
        idx = self.new_index()
        idx.sync(self.collection)
        with gzip.open(self.path, "rb") as f:
            payload = json.loads(f.read().decode("utf-8"))
        self.assertNotIn("The defaulter may request", json.dumps(payload))
        self.assertEqual(set(payload["docs"]), {"a", "b", "c"})

    def test_tokenizer_change_forces_rebuild(self):
        self.new_index().sync(self.collection)
        idx = self.new_index(Tokenizer(subword="ngram"))
        self.assertTrue(idx.sync(self.collection).rebuilt)

    def test_corrupt_file_rebuilds(self):
        with open(self.path, "wb") as f:
            f.write(b"not gzip")
        idx = self.new_index()
        stats = idx.sync(self.collection)
        self.assertTrue(stats.rebuilt)
        self.assertEqual(len(idx), 3)

    def test_pagination(self):
        many = make_collection([(f"id{i:04d}", f"word{i} common") for i in range(1203)])
        idx = self.new_index()
        self.assertEqual(idx.sync(many).total, 1203)
        self.assertEqual(idx.search("word1202", 1)[0][0], "id1202")


if __name__ == "__main__":
    unittest.main()
