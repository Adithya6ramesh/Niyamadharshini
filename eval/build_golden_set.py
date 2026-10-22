"""
Resolve and verify the golden evaluation set against the indexed corpus.

Each answerable question lists verbatim `evidence` passages copied from the
indexed chunks. This script finds every chunk containing each passage (after
the same Unicode normalization BM25 uses, so ZWJ/chillu variants still match)
and records the resulting gold chunk IDs and PDF pages. Gold IDs are therefore
derived from the corpus, never typed by hand.

    python eval/build_golden_set.py           # verify only (exit 1 on drift)
    python eval/build_golden_set.py --write   # (re)write gold_chunk_ids / gold_pages
    python eval/build_golden_set.py --show    # print matched chunk text for review

Fails if an evidence passage matches nothing, or if an unanswerable question's
absence probe matches any chunk.
"""

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
from chromadb import PersistentClient

from retrieval import RetrievalConfig
from retrieval.tokenizer import normalize

GOLDEN_PATH = os.path.join(ROOT, "eval", "golden_set.jsonl")
DERIVED = ("gold_chunk_ids", "gold_pages")


def _canon(text: str) -> str:
    return " ".join(normalize(text).split())


def load_golden(path=GOLDEN_PATH):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(collection):
    corpus = {}
    offset = 0
    while True:
        page = collection.get(include=["documents", "metadatas"], limit=500, offset=offset)
        if not page["ids"]:
            break
        for cid, doc, meta in zip(page["ids"], page["documents"], page["metadatas"]):
            corpus[cid] = {"text": doc or "", "canon": _canon(doc or ""), "meta": meta or {}}
        offset += len(page["ids"])
    return corpus


def resolve(question, corpus):
    errors = []
    gold = []
    for ev in question.get("evidence", []):
        needle = _canon(ev["text"])
        hits = sorted(
            cid for cid, c in corpus.items()
            if needle in c["canon"] and (not ev.get("source") or c["meta"].get("source") == ev["source"])
        )
        if not hits:
            errors.append(f"{question['id']}: evidence not found: {ev['text'][:60]!r}")
        ev["chunk_ids"] = hits
        gold.extend(h for h in hits if h not in gold)

    probe = question.get("absence_probe")
    if probe:
        pattern = re.compile(probe, re.I)
        found = [cid for cid, c in corpus.items() if pattern.search(c["text"])]
        if found:
            errors.append(f"{question['id']}: absence probe {probe!r} matched {found[:5]}")
    if question["answerable"] and not gold:
        errors.append(f"{question['id']}: answerable question has no gold chunks")

    pages = sorted({(corpus[c]["meta"].get("source"), str(corpus[c]["meta"].get("page"))) for c in gold})
    return gold, [{"pdf": s, "page": p} for s, p in pages], errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv()
    config = RetrievalConfig.from_env()
    collection = PersistentClient(path=config.chroma_path).get_collection(config.collection_name)
    corpus = load_corpus(collection)
    questions = load_golden()

    all_errors, drift = [], []
    for q in questions:
        gold, pages, errors = resolve(q, corpus)
        all_errors.extend(errors)
        if q.get("gold_chunk_ids") != gold or q.get("gold_pages") != pages:
            drift.append(q["id"])
        q["gold_chunk_ids"], q["gold_pages"] = gold, pages
        print(f"{q['id']} [{q['language']:5}] {q['category']:12} gold={len(gold):2d} "
              f"pages={', '.join(p['pdf'] + ':' + p['page'] for p in pages) or '-'}")
        if args.show:
            for cid in gold:
                print(f"    - {cid}: {corpus[cid]['text'][:220]!r}")

    for e in all_errors:
        print("ERROR", e)
    if args.write and not all_errors:
        with open(GOLDEN_PATH, "w", encoding="utf-8", newline="\n") as f:
            for q in questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"Wrote {len(questions)} questions to {GOLDEN_PATH}")
    elif drift:
        print(f"Derived gold fields out of date for: {', '.join(drift)} (run with --write)")
    sys.exit(1 if all_errors or (drift and not args.write) else 0)


if __name__ == "__main__":
    main()
