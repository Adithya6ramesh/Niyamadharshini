"""
Export the evaluation results for the report page (docs/index.html).

Collects every saved run in eval/results/ and the golden set into
docs/data.json, keeping only the fields the page reads. The docs/ folder is
then a complete static site: serve it with eval/serve_report.py, through the
app at /evals, or deploy it as-is (Vercel, Netlify, GitHub Pages).

    python eval/build_report_data.py

evaluate_retrieval.py runs this automatically after every evaluation.
"""

import json
import os
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "eval", "results")
GOLDEN_PATH = os.path.join(ROOT, "eval", "golden_set.jsonl")
OUT_PATH = os.path.join(ROOT, "docs", "data.json")

# Only the fields the page reads, to keep the file small
HIT_FIELDS = ("chunk_id", "pdf", "page", "rerank_score")
ANSWER_FIELDS = ("answer", "error", "model", "generation_ms", "ranked_chunk_ids")


def _slim_retrieval(retrieval):
    configs = {}
    for name, per_q in retrieval["configs"].items():
        configs[name] = {
            qid: {
                "ranked_chunk_ids": e["ranked_chunk_ids"],
                "timings_ms": e.get("timings_ms", {}),
                "metrics": e.get("metrics"),
                "hits": [{k: h.get(k) for k in HIT_FIELDS} for h in e.get("hits", [])],
            }
            for qid, e in per_q.items()
        }
    return {"config": retrieval.get("config", {}), "configs": configs}


def _slim_answers(answers):
    if not answers:
        return None
    return {cfg: {qid: {k: a.get(k) for k in ANSWER_FIELDS} for qid, a in per_q.items()}
            for cfg, per_q in answers.items()}


def collect():
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = [json.loads(line) for line in f if line.strip()]
    runs = []
    for name in sorted(os.listdir(RESULTS_DIR)) if os.path.isdir(RESULTS_DIR) else []:
        run_dir = os.path.join(RESULTS_DIR, name)
        retrieval_path = os.path.join(run_dir, "retrieval.json")
        if not os.path.isfile(retrieval_path):
            continue
        with open(retrieval_path, encoding="utf-8") as f:
            retrieval = json.load(f)
        answers = None
        answers_path = os.path.join(run_dir, "answers.json")
        if os.path.isfile(answers_path):
            with open(answers_path, encoding="utf-8") as f:
                answers = json.load(f)
        runs.append({
            "name": name,
            "modified": os.path.getmtime(retrieval_path),
            "retrieval": _slim_retrieval(retrieval),
            "answers": _slim_answers(answers),
        })
    runs.sort(key=lambda r: r["modified"], reverse=True)
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "golden": golden, "runs": runs}


def build(out_path=OUT_PATH):
    data = collect()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    return out_path, len(data["runs"]), os.path.getsize(out_path)


if __name__ == "__main__":
    path, n_runs, size = build()
    print(f"Wrote {path} ({n_runs} runs, {size / 1024:.0f} KB)")
