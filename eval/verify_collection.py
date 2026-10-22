"""
Fingerprint a ChromaDB collection to prove the retrieval upgrade left it unchanged.

Hashes every record's ID, document text, metadata and embedding bytes (in ID
order), plus the collection's configuration. Read-only.

    python eval/verify_collection.py                       # print fingerprint of CHROMA_DB_PATH
    python eval/verify_collection.py --save baseline.json  # store it
    python eval/verify_collection.py --compare baseline.json   # exit 1 if anything differs
    python eval/verify_collection.py --path other_chroma_dir   # fingerprint another copy
"""

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from dotenv import load_dotenv
from chromadb import PersistentClient

from retrieval import RetrievalConfig


def fingerprint(path: str, collection_name: str) -> dict:
    col = PersistentClient(path=path).get_collection(collection_name)
    records = {}
    offset = 0
    while True:
        page = col.get(include=["documents", "metadatas", "embeddings"], limit=500, offset=offset)
        if not page["ids"]:
            break
        for i, cid in enumerate(page["ids"]):
            records[cid] = (page["documents"][i] or "", page["metadatas"][i] or {},
                            np.asarray(page["embeddings"][i], dtype=np.float32).tobytes())
        offset += len(page["ids"])

    h_docs, h_meta, h_emb, h_all = (hashlib.sha256() for _ in range(4))
    for cid in sorted(records):
        doc, meta, emb = records[cid]
        meta_json = json.dumps(meta, sort_keys=True, ensure_ascii=False).encode("utf-8")
        h_docs.update(cid.encode() + b"\0" + doc.encode("utf-8") + b"\0")
        h_meta.update(cid.encode() + b"\0" + meta_json + b"\0")
        h_emb.update(cid.encode() + b"\0" + emb)
        h_all.update(cid.encode() + b"\0" + doc.encode("utf-8") + b"\0" + meta_json + b"\0" + emb)

    ids_hash = hashlib.sha256("\n".join(sorted(records)).encode()).hexdigest()
    return {
        "collection": collection_name,
        "count": len(records),
        "configuration": json.loads(json.dumps(col.configuration_json, default=str)),
        "metadata": col.metadata,
        "sha256_ids": ids_hash,
        "sha256_documents": h_docs.hexdigest(),
        "sha256_metadatas": h_meta.hexdigest(),
        "sha256_embeddings": h_emb.hexdigest(),
        "sha256_all": h_all.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path")
    parser.add_argument("--save")
    parser.add_argument("--compare")
    args = parser.parse_args()

    load_dotenv()
    cfg = RetrievalConfig.from_env()
    fp = fingerprint(args.path or cfg.chroma_path, cfg.collection_name)
    print(json.dumps(fp, indent=1))

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(fp, f, indent=1)
    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            base = json.load(f)
        diffs = [k for k in fp if fp[k] != base.get(k)]
        print("IDENTICAL" if not diffs else f"DIFFERENT: {', '.join(diffs)}")
        sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
