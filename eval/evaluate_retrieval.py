"""
Golden-set evaluation: A (original dense) vs B (hybrid RRF) vs C (hybrid + rerank).

Read-only with respect to ChromaDB; the BM25 index used here is built in a
temporary directory, never at the production index path.

    python eval/evaluate_retrieval.py                  # retrieval metrics for A/B/C
    python eval/evaluate_retrieval.py --generate       # + answers from the production prompt/model
    python eval/evaluate_retrieval.py --sweeps         # + tokenizer, depth/RRF-k, rerank-depth sweeps
    python eval/evaluate_retrieval.py --score-review eval/results/<run>/manual_review.csv

Each run writes eval/results/<timestamp>/:
    retrieval.json      ranked chunk IDs, citations, stage latencies, metrics per question
    answers.json        generated answers per config (with --generate; resumable via --resume)
    manual_review.csv   answer-quality sheet to be filled in by a human reviewer
    summary.md          the metric tables printed below

Retrieval metrics (answerable questions, top-10 context sent to the LLM):
    Hit@k       a gold chunk appears in the top k
    Recall@k    gold chunks retrieved / gold chunks
    P@k         gold chunks in top k / k
    MRR@10      reciprocal rank of the first gold chunk
    nDCG@10     binary-relevance nDCG
    Complete@10 evidence facts covered / facts required (a fact counts as covered if any
                chunk holding it, in either language, is in the top 10)
Answer quality (correctness, faithfulness, citation support, abstention) is judged
manually from manual_review.csv; this script never scores answers automatically.
"""

import argparse
import ast
import csv
import json
import math
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

from retrieval import BM25Index, CrossEncoderReranker, HybridRetriever, RetrievalConfig, Tokenizer
from retrieval.fusion import reciprocal_rank_fusion

GOLDEN_PATH = os.path.join(ROOT, "eval", "golden_set.jsonl")
TOP_K = 10
CONFIGS = ("A_dense_baseline", "B_hybrid", "C_hybrid_rerank")
REVIEW_FIELDS = ("answer_correct", "faithful", "citations_support_claims", "abstained_correctly")


# ---------------------------------------------------------------- metrics

def question_metrics(q, ranked, page_of):
    gold = set(q["gold_chunk_ids"])
    top = ranked[:TOP_K]
    first = next((i for i, c in enumerate(top, 1) if c in gold), None)
    facts = {}
    for ev in q["evidence"]:
        facts.setdefault(ev.get("fact", ev["text"]), set()).update(ev.get("chunk_ids", []))
    covered = sum(1 for ids in facts.values() if ids & set(top))
    gold_pages = {(p["pdf"], p["page"]) for p in q["gold_pages"]}
    dcg = sum(1 / math.log2(i + 1) for i, c in enumerate(top, 1) if c in gold)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), TOP_K) + 1))
    return {
        "hit@1": int(first == 1), "hit@5": int(first is not None and first <= 5), "hit@10": int(first is not None),
        "page_hit@5": int(any(page_of.get(c) in gold_pages for c in top[:5])),
        "recall@5": len(gold & set(top[:5])) / len(gold), "recall@10": len(gold & set(top)) / len(gold),
        "p@5": len(gold & set(top[:5])) / 5, "p@10": len(gold & set(top)) / TOP_K,
        "mrr@10": 1 / first if first else 0.0, "ndcg@10": dcg / idcg if idcg else 0.0,
        "facts_covered": covered, "facts_total": len(facts),
        "first_gold_rank": first,
    }


def summarize(per_q):
    n = len(per_q)
    s = {"n": n}
    for k in ("hit@1", "hit@5", "hit@10", "page_hit@5"):
        s[k] = sum(m[k] for m in per_q)
    for k in ("recall@5", "recall@10", "p@5", "p@10", "mrr@10", "ndcg@10"):
        s[k] = sum(m[k] for m in per_q) / n if n else 0.0
    s["facts_covered"] = sum(m["facts_covered"] for m in per_q)
    s["facts_total"] = sum(m["facts_total"] for m in per_q)
    s["fully_complete"] = sum(m["facts_covered"] == m["facts_total"] for m in per_q)
    return s


def table(rows, title):
    out = [f"\n### {title}\n",
           "| Config | n | Hit@1 | Hit@5 | Hit@10 | PageHit@5 | Recall@5 | Recall@10 | P@5 | P@10 | MRR@10 | nDCG@10 | Facts covered | Complete Qs |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in rows:
        n = s["n"]
        c = lambda k: f"{s[k]}/{n} ({100 * s[k] / n:.0f}%)" if n else "-"
        out.append(f"| {name} | {n} | {c('hit@1')} | {c('hit@5')} | {c('hit@10')} | {c('page_hit@5')} | "
                   f"{s['recall@5']:.3f} | {s['recall@10']:.3f} | {s['p@5']:.3f} | {s['p@10']:.3f} | "
                   f"{s['mrr@10']:.3f} | {s['ndcg@10']:.3f} | {s['facts_covered']}/{s['facts_total']} | "
                   f"{s['fully_complete']}/{n} |")
    return "\n".join(out)


# ---------------------------------------------------------------- production prompt / model

API_GENERATION_NAMES = ("build_prompt", "DEFAULT_LLM_MODEL", "LLM_MAX_TOKENS", "LLM_FALLBACK_BETA",
                        "RefusalError", "get_llm_model", "make_llm_client", "generate_answer")


def load_from_api(names=API_GENERATION_NAMES):
    """
    Load the prompt and Claude generation code from api.py without running the
    module (importing api.py would load the embedding and reranker models), so
    the eval uses exactly the production prompt, model and request settings.
    """
    import anthropic
    from types import SimpleNamespace
    path = os.path.join(ROOT, "api.py")
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    nodes = [n for n in tree.body
             if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    ns = {"os": os, "anthropic": anthropic}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
    missing = [n for n in names if n not in ns]
    if missing:
        raise RuntimeError(f"api.py no longer defines: {', '.join(missing)}")
    return SimpleNamespace(**{n: ns[n] for n in names})


def generate(api, client, prompt):
    """Same call as the app (api.generate_answer); the SDK retries 429/5xx/overloaded."""
    import anthropic
    start = time.perf_counter()
    try:
        answer = api.generate_answer(client, prompt)
        return answer, (time.perf_counter() - start) * 1000, None
    except api.RefusalError as e:
        return None, None, f"refusal ({e.category})"
    except anthropic.APIStatusError as e:
        return None, None, f"{e.status_code} {type(e).__name__}"
    except anthropic.APIError as e:  # connection errors / timeouts
        return None, None, type(e).__name__


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generate", action="store_true", help="also generate answers with the production prompt")
    parser.add_argument("--sweeps", action="store_true", help="tokenizer / depth / RRF-k / rerank-depth sweeps")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--resume", default=None, help="existing run dir: reuse its answers.json")
    parser.add_argument("--gen-interval", type=float, default=1.0, help="seconds between LLM calls")
    parser.add_argument("--score-review", default=None, help="summarize a filled manual_review.csv and exit")
    parser.add_argument("--generate-only", default=None, metavar="RUN_DIR",
                        help="generate answers for an existing run's ranked chunks (no models loaded) and exit")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.score_review:
        score_review(args.score_review)
        return
    if args.generate_only:
        generate_for_existing_run(args, args.generate_only)
        return

    load_dotenv()
    cfg = RetrievalConfig.from_env()
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    answerable = [q for q in questions if q["answerable"]]
    out_dir = args.resume or args.out_dir or os.path.join(ROOT, "eval", "results", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer

    print(f"[eval] {len(questions)} questions ({len(answerable)} answerable) -> {out_dir}")
    collection = PersistentClient(path=cfg.chroma_path).get_collection(cfg.collection_name)
    allmeta = collection.get(include=["metadatas"])
    page_of = {cid: (m.get("source"), str(m.get("page"))) for cid, m in zip(allmeta["ids"], allmeta["metadatas"])}

    print(f"[eval] loading {cfg.embed_model_name}")
    embed_model = SentenceTransformer(cfg.embed_model_name)
    tmp = tempfile.mkdtemp(prefix="bm25-eval-")
    bm25 = BM25Index(os.path.join(tmp, "prod.json.gz"), Tokenizer(), k1=cfg.bm25_k1, b=cfg.bm25_b,
                     collection_name=cfg.collection_name)
    bm25.rebuild(collection, save=False)

    reranker = None
    if cfg.rerank_enabled:
        reranker = CrossEncoderReranker(cfg.reranker_model, max_length=cfg.rerank_max_length,
                                        batch_size=cfg.rerank_batch_size, device=cfg.reranker_device,
                                        quantize=cfg.rerank_quantize)
        if not reranker.load():
            print(f"[eval] reranker unavailable ({reranker.load_error}); config C will be skipped")
            reranker = None
    retriever = HybridRetriever(cfg, collection, embed_model, bm25_index=bm25, reranker=reranker)

    # ---- the three configurations, same questions, same top-10 context budget ----
    def run_a(query):
        """Original api.py retrieve_context(), verbatim."""
        t = {}
        s = time.perf_counter()
        emb = embed_model.encode([query])[0].tolist()
        t["embed"] = (time.perf_counter() - s) * 1000
        s = time.perf_counter()
        res = collection.query(query_embeddings=[emb], n_results=TOP_K, include=["documents", "metadatas", "distances"])
        t["dense"] = (time.perf_counter() - s) * 1000
        t["total"] = t["embed"] + t["dense"]
        hits = [{"chunk_id": cid, "text": d, "pdf": m.get("source"), "page": m.get("page"), "distance": dist}
                for cid, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]
        return hits, t, {}

    def run_retriever(rerank):
        def run(query):
            r = retriever.retrieve(query, top_k=TOP_K, mode="hybrid", rerank=rerank)
            hits = [{"chunk_id": c.chunk_id, "text": c.text, "pdf": c.metadata.get("source"),
                     "page": c.metadata.get("page"), "distance": c.distance, "dense_rank": c.dense_rank,
                     "bm25_rank": c.bm25_rank, "rerank_score": c.rerank_score} for c in r.candidates]
            return hits, dict(r.timings_ms), {"reranked": r.reranked, "warnings": r.warnings}
        return run

    runners = {"A_dense_baseline": run_a, "B_hybrid": run_retriever(False)}
    if reranker is not None:
        runners["C_hybrid_rerank"] = run_retriever(True)

    results = {"config": {k: v for k, v in cfg.__dict__.items()}, "configs": {}}
    run_a("warmup")  # first call includes one-off model warmup; keep it out of latency numbers
    for name, run in runners.items():
        print(f"[eval] running {name}")
        per_q = {}
        for q in questions:
            hits, timings, info = run(q["query"])
            ranked = [h["chunk_id"] for h in hits]
            entry = {"query": q["query"], "ranked_chunk_ids": ranked, "timings_ms": timings, **info,
                     "citations": [{"pdf": h["pdf"], "page": h["page"], "chunk_id": h["chunk_id"]} for h in hits],
                     "hits": [{k: v for k, v in h.items() if k != "text"} for h in hits],
                     "_contexts": [h["text"] for h in hits]}
            if q["answerable"]:
                entry["metrics"] = question_metrics(q, ranked, page_of)
            per_q[q["id"]] = entry
        results["configs"][name] = per_q

    # ---- report ----
    lines = [f"# Retrieval evaluation ({datetime.now():%Y-%m-%d %H:%M})",
             f"\nGolden set: {len(questions)} questions, {len(answerable)} answerable, "
             f"{len(questions) - len(answerable)} unanswerable. Small benchmark: differences of one or two "
             f"questions are within noise.",
             f"\nC settings: reranker={cfg.reranker_model}, candidates={cfg.rerank_candidates}, "
             f"max_length={cfg.rerank_max_length}, quantize={cfg.rerank_quantize or 'none'}; "
             f"B/C: dense_k={cfg.dense_k}, bm25_k={cfg.bm25_k}, rrf_k={cfg.rrf_k}."]
    summaries = {}
    for name, per_q in results["configs"].items():
        summaries[name] = summarize([per_q[q["id"]]["metrics"] for q in answerable])
    lines.append(table(list(summaries.items()), "All answerable questions"))
    for group, key in (("category", "category"), ("language", "language")):
        for value in sorted({q[key] for q in answerable}):
            qs = [q for q in answerable if q[key] == value]
            lines.append(table([(name, summarize([per_q[q["id"]]["metrics"] for q in qs]))
                                for name, per_q in results["configs"].items()], f"By {group}: {value}"))

    lines.append("\n### First gold rank per question (- = not in top 10)\n")
    lines.append("| Q | lang | category | " + " | ".join(results["configs"]) + " |")
    lines.append("|---|---|---|" + "---|" * len(results["configs"]))
    for q in answerable:
        ranks = [str(per_q[q["id"]]["metrics"]["first_gold_rank"] or "-") for per_q in results["configs"].values()]
        lines.append(f"| {q['id']} | {q['language']} | {q['category']} | " + " | ".join(ranks) + " |")

    lines.append("\n### Latency per question (ms, median / max)\n")
    lines.append("| Config | stage | median | max |\n|---|---|---|---|")
    for name, per_q in results["configs"].items():
        stages = {}
        for e in per_q.values():
            for k, v in e["timings_ms"].items():
                stages.setdefault(k, []).append(v)
        for k, v in stages.items():
            lines.append(f"| {name} | {k} | {statistics.median(v):.0f} | {max(v):.0f} |")

    if "C_hybrid_rerank" in results["configs"]:
        c = results["configs"]["C_hybrid_rerank"]
        top = lambda qid: max((h["rerank_score"] or 0) for h in c[qid]["hits"]) if c[qid]["hits"] else 0
        ans = sorted(top(q["id"]) for q in answerable)
        una = {q["id"]: round(top(q["id"]), 4) for q in questions if not q["answerable"]}
        lines.append(f"\n### Top rerank score\n\nAnswerable: min {ans[0]:.4f}, median {statistics.median(ans):.4f}, "
                     f"max {ans[-1]:.4f}. Unanswerable: {una}.")

    if args.sweeps:
        lines.append(run_sweeps(cfg, questions, answerable, collection, embed_model, retriever, page_of))

    # ---- generation ----
    if args.generate:
        lines.append(run_generation(args, out_dir, questions, results))

    write_review_sheet(out_dir, questions, results)
    for per_q in results["configs"].values():
        for e in per_q.values():
            e.pop("_contexts", None)
    with open(os.path.join(out_dir, "retrieval.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    report = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\n[eval] wrote {out_dir}")


def run_sweeps(cfg, questions, answerable, collection, embed_model, retriever, page_of):
    """Diagnostics on cached ranked lists (cheap): which tokenizer / depth / k / rerank depth works best."""
    depth = 50
    dense = {}
    for q in questions:
        emb = embed_model.encode([q["query"]])[0].tolist()
        dense[q["id"]] = collection.query(query_embeddings=[emb], n_results=depth, include=[])["ids"][0]
    tmp = tempfile.mkdtemp(prefix="bm25-sweep-")
    variants = {"raw-whitespace": _RawWhitespaceTokenizer(), "norm-only": Tokenizer(subword="none"),
                "prefix4": Tokenizer(prefix_len=4), "prefix5": Tokenizer(prefix_len=5),
                "prefix6": Tokenizer(prefix_len=6), "ngram3": Tokenizer(subword="ngram", ngram_n=3),
                "ngram4": Tokenizer(subword="ngram", ngram_n=4)}
    bm25 = {}
    for name, tok in variants.items():
        idx = BM25Index(os.path.join(tmp, name), tok, k1=cfg.bm25_k1, b=cfg.bm25_b, collection_name=cfg.collection_name)
        idx.rebuild(collection, save=False)
        bm25[name] = {q["id"]: [c for c, _ in idx.search(q["query"], depth)] for q in questions}

    def evaluate(rank_fn):
        return summarize([question_metrics(q, rank_fn(q), page_of) for q in answerable])

    rows = [(f"BM25 only, {n}", evaluate(lambda q, n=n: bm25[n][q["id"]])) for n in variants]
    rows.append(("Dense only", evaluate(lambda q: dense[q["id"]])))
    out = [table(rows, "Sweep: BM25 tokenizer (BM25 alone)")]

    prod = Tokenizer().version.split("|")[1]
    rows = []
    for d in (10, 20, 30, 50):
        for k in (10, 30, 60, 100):
            rows.append((f"hybrid {prod} depth={d} rrf_k={k}", evaluate(
                lambda q, d=d, k=k: [i.chunk_id for i in reciprocal_rank_fusion(
                    {"dense": dense[q["id"]][:d], "bm25": bm25[prod][q["id"]][:d]}, k=k)])))
    for n in variants:
        rows.append((f"hybrid {n} depth={cfg.dense_k} rrf_k={cfg.rrf_k}", evaluate(
            lambda q, n=n: [i.chunk_id for i in reciprocal_rank_fusion(
                {"dense": dense[q["id"]][:cfg.dense_k], "bm25": bm25[n][q["id"]][:cfg.bm25_k]}, k=cfg.rrf_k)])))
    out.append(table(rows, "Sweep: hybrid depth / RRF k / tokenizer"))

    if retriever.reranker is not None:
        rows = []
        for n in (10, 20, 30):
            cfg_n = RetrievalConfig(**{**cfg.__dict__, "rerank_candidates": n})
            r = HybridRetriever(cfg_n, collection, embed_model, retriever.bm25_index, retriever.reranker)
            rows.append((f"hybrid + rerank top{n}", evaluate(
                lambda q, r=r: [c.chunk_id for c in r.retrieve(q["query"], top_k=TOP_K, rerank=True).candidates])))
        out.append(table(rows, "Sweep: rerank candidate depth"))
    return "\n".join(out)


def run_generation(args, out_dir, questions, results):
    api = load_from_api()
    build_prompt = api.build_prompt
    model = api.get_llm_model()
    try:
        client = api.make_llm_client()
    except RuntimeError as e:
        return f"\n### Generation\n\nSkipped: {e}"
    path = os.path.join(out_dir, "answers.json")
    answers = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            answers = json.load(f)
    failures = 0
    for name, per_q in results["configs"].items():
        for q in questions:
            prev = answers.get(name, {}).get(q["id"])
            if (prev and prev.get("answer") and prev.get("model") == model
                    and prev.get("ranked_chunk_ids") == per_q[q["id"]]["ranked_chunk_ids"]):
                continue  # resume: same model and same context already answered
            prompt = build_prompt(q["query"], per_q[q["id"]]["_contexts"])
            answer, ms, err = generate(api, client, prompt)
            failures += err is not None
            answers.setdefault(name, {})[q["id"]] = {
                "answer": answer, "error": err, "generation_ms": ms, "model": model,
                "ranked_chunk_ids": per_q[q["id"]]["ranked_chunk_ids"],
                "citations": per_q[q["id"]]["citations"]}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(answers, f, ensure_ascii=False, indent=1)
            print(f"[gen] {name} {q['id']}: {'ok' if answer else err}")
            time.sleep(args.gen_interval)
    results["answers"] = answers
    done = sum(1 for c in answers.values() for a in c.values() if a.get("answer"))
    return (f"\n### Generation\n\nModel {model}, production prompt from api.py, same top-{TOP_K} context budget "
            f"per config. Answers generated: {done}; failed this run: {failures} (re-run with --resume to retry). "
            f"Answer quality is scored manually in manual_review.csv.")


def generate_for_existing_run(args, run_dir):
    """Answer generation for a saved run: same ranked chunks, texts re-read from ChromaDB by ID."""
    from chromadb import PersistentClient
    load_dotenv()
    cfg = RetrievalConfig.from_env()
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    path = os.path.join(run_dir, "retrieval.json")
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    collection = PersistentClient(path=cfg.chroma_path).get_collection(cfg.collection_name)
    for per_q in results["configs"].values():
        for e in per_q.values():
            ids = e["ranked_chunk_ids"]
            got = collection.get(ids=ids, include=["documents"]) if ids else {"ids": [], "documents": []}
            text = dict(zip(got["ids"], got["documents"]))
            missing = [i for i in ids if i not in text]
            if missing:
                raise SystemExit(f"Chunks missing from ChromaDB (collection changed since the run?): {missing[:3]}")
            e["_contexts"] = [text[i] for i in ids]
    summary = run_generation(args, run_dir, questions, results)
    write_review_sheet(run_dir, questions, results)
    md = os.path.join(run_dir, "summary.md")
    if os.path.exists(md):
        with open(md, encoding="utf-8") as f:
            existing = f.read()
        existing = existing.split("\n### Generation\n")[0].rstrip()
        with open(md, "w", encoding="utf-8") as f:
            f.write(existing + "\n" + summary + "\n")
    print(summary)


def write_review_sheet(out_dir, questions, results):
    answers = results.get("answers", {})
    path = os.path.join(out_dir, "manual_review.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "config", "language", "category", "answerable", "query", "expected_answer",
                    "gold_pages", "answer", "citations", *REVIEW_FIELDS, "reviewer_notes"])
        for q in questions:
            for name, per_q in results["configs"].items():
                a = answers.get(name, {}).get(q["id"], {})
                w.writerow([q["id"], name, q["language"], q["category"], q["answerable"], q["query"],
                            q["expected_answer"], "; ".join(f"{p['pdf']} p{p['page']}" for p in q["gold_pages"]),
                            a.get("answer") or "", "; ".join(f"{c['pdf']} p{c['page']}" for c in per_q[q["id"]]["citations"]),
                            "", "", "", "" if q["answerable"] else "", ""])


def score_review(path):
    """Summarize a manually filled review sheet (values: y / n / partial; blank = not reviewed)."""
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_config = {}
    for r in rows:
        by_config.setdefault(r["config"], []).append(r)
    print("| Config | " + " | ".join(REVIEW_FIELDS) + " |\n|---|" + "---|" * len(REVIEW_FIELDS))
    for name, rs in by_config.items():
        cells = []
        for field in REVIEW_FIELDS:
            scope = [r for r in rs if (field == "abstained_correctly") == (r["answerable"] == "False")]
            vals = [r[field].strip().lower() for r in scope if r[field].strip()]
            yes = sum(v in ("y", "yes", "1", "true") for v in vals)
            part = sum(v.startswith("p") for v in vals)
            cells.append(f"{yes}/{len(vals)} yes, {part} partial ({len(scope) - len(vals)} unreviewed)")
        print(f"| {name} | " + " | ".join(cells) + " |")


class _RawWhitespaceTokenizer:
    """Baseline for comparison only: what a naive whitespace BM25 would do."""
    version = "raw-whitespace"

    def tokenize(self, text):
        return (text or "").lower().split()


if __name__ == "__main__":
    main()
