"""
Build or synchronize the BM25 lexical index from the existing ChromaDB collection.

Reads chunk texts from ChromaDB (read-only); never re-processes PDFs or touches
embeddings. Run after ingestion, or any time the index file is lost:

    python scripts/build_bm25_index.py            # incremental sync (default)
    python scripts/build_bm25_index.py --rebuild  # rebuild from scratch
    python scripts/build_bm25_index.py --check    # report drift without writing
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from chromadb import PersistentClient

from retrieval import BM25Index, RetrievalConfig, Tokenizer
from retrieval.bm25_index import _iter_collection, content_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rebuild", action="store_true", help="discard the index and rebuild it")
    group.add_argument("--check", action="store_true", help="report differences only; do not write")
    args = parser.parse_args()

    load_dotenv()
    config = RetrievalConfig.from_env()
    collection = PersistentClient(path=config.chroma_path).get_collection(config.collection_name)
    index = BM25Index(config.bm25_index_path, Tokenizer(), k1=config.bm25_k1, b=config.bm25_b,
                      collection_name=config.collection_name)

    print(f"Collection: {config.collection_name} ({collection.count()} chunks) at {config.chroma_path}")
    print(f"Index file: {config.bm25_index_path}")
    print(f"Tokenizer:  {index.tokenizer.version}")

    if args.check:
        if not index.load():
            print("Index missing or stale: a full rebuild is required.")
            sys.exit(1)
        indexed = {cid: doc["h"] for cid, doc in index._docs.items()}
        current = {cid: content_hash(text) for ids, docs in _iter_collection(collection)
                   for cid, text in zip(ids, docs)}
        added = current.keys() - indexed.keys()
        removed = indexed.keys() - current.keys()
        updated = {c for c in current.keys() & indexed.keys() if current[c] != indexed[c]}
        print(f"In sync: {not (added or removed or updated)}  "
              f"to_add={len(added)} to_remove={len(removed)} to_update={len(updated)}")
        sys.exit(0 if not (added or removed or updated) else 1)

    stats = index.rebuild(collection) if args.rebuild else index.sync(collection)
    print(f"Done: total={stats.total} added={stats.added} updated={stats.updated} "
          f"removed={stats.removed} rebuilt={stats.rebuilt} in {stats.seconds:.2f}s")


if __name__ == "__main__":
    main()
