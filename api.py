"""
Flask API and web server for Niyamadharshini.

Question -> hybrid retrieval (retrieval/) -> Claude answer with page citations.
Also serves the chat UI (/), the source PDFs (/data/...) and the retrieval
evaluation viewer (/evals).
"""

from flask import Flask, Response, request, jsonify, send_from_directory
from flask_cors import CORS
import hmac
import json
import logging
import os
from dotenv import load_dotenv
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import anthropic

from retrieval import RetrievalConfig, build_retriever

# Load environment
load_dotenv(override=True)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("retrieval").setLevel(logging.INFO)

app = Flask(__name__)

# ---------- Access control ----------
# The UI is served by this app (same origin), so CORS is only needed for other front-ends;
# list them explicitly in CORS_ORIGINS (comma-separated). Default: none.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if CORS_ORIGINS:
    CORS(app, origins=CORS_ORIGINS)

# Optional password for every page, API route and PDF (HTTP Basic; any username).
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
AUTH_EXEMPT_PATHS = {"/api/health"}


@app.before_request
def require_password():
    if not APP_PASSWORD or request.path in AUTH_EXEMPT_PATHS:
        return None
    auth = request.authorization
    supplied = (auth.password or "") if auth else ""
    if hmac.compare_digest(supplied.encode("utf-8"), APP_PASSWORD.encode("utf-8")):
        return None
    return Response("Authentication required", 401, {"WWW-Authenticate": 'Basic realm="Niyamadharshini"'})


# ---------- Answer generation: Claude (Anthropic API) ----------
# .env: ANTHROPIC_API_KEY (required), LLM_MODEL (optional, default claude-sonnet-5-5)

DEFAULT_LLM_MODEL = "claude-sonnet-5-5"
LLM_MAX_TOKENS = 16000
# Server-side refusal fallback: if a safety classifier declines, the API retries
# on Anthropic's recommended fallback model inside the same call.
LLM_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class RefusalError(RuntimeError):
    """The model (and its fallback) declined the request."""

    def __init__(self, category):
        super().__init__(f"Request declined by the model (category: {category or 'unspecified'})")
        self.category = category


def get_llm_model() -> str:
    return os.getenv("LLM_MODEL", "").strip() or DEFAULT_LLM_MODEL


def make_llm_client() -> anthropic.Anthropic:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not found. Add it to the .env file in the project root.")
    # The SDK retries 408/409/429/5xx (including 529 overloaded) and connection
    # errors with exponential backoff; timeout is per attempt, in seconds.
    return anthropic.Anthropic(timeout=120.0, max_retries=3)


def generate_answer(client: anthropic.Anthropic, prompt: str) -> str:
    response = client.beta.messages.create(
        model=get_llm_model(),
        max_tokens=LLM_MAX_TOKENS,
        betas=[LLM_FALLBACK_BETA],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        raise RefusalError(response.stop_details.category if response.stop_details else None)
    return "".join(block.text for block in response.content if block.type == "text").strip()


llm_client = make_llm_client()

RETRIEVAL_CONFIG = RetrievalConfig.from_env()
CHROMA_PATH = RETRIEVAL_CONFIG.chroma_path
COLLECTION_NAME = RETRIEVAL_CONFIG.collection_name
EMBED_MODEL_NAME = RETRIEVAL_CONFIG.embed_model_name
TOP_K = RETRIEVAL_CONFIG.final_top_k

# Initialize models (load once at startup)
print("[INIT] Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

print("[INIT] Connecting to ChromaDB...")
chroma_client = PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)

print(f"[INIT] Preparing retriever (mode={RETRIEVAL_CONFIG.mode}, rerank={RETRIEVAL_CONFIG.rerank_enabled})...")
retriever = build_retriever(RETRIEVAL_CONFIG, embed_model, collection)

print(f"[INIT] Answer generation: Claude ({get_llm_model()})")


def retrieve_context(query: str, top_k: int = TOP_K):
    """Retrieve relevant chunks (dense + BM25 -> RRF -> rerank, per RETRIEVAL_* settings)."""
    candidates = retriever.retrieve(query, top_k=top_k).candidates

    # Every hit normally has a Chroma distance; fall back to the worst known one so the
    # response never carries a null distance/relevance.
    known = [c.distance for c in candidates if c.distance is not None]
    fallback = max(known) if known else 1.0

    contexts = []
    sources = []

    for cand in candidates:
        meta = cand.metadata or {}
        distance = cand.distance if cand.distance is not None else fallback
        # Displayed relevance = embedding cosine similarity (Chroma "l2" on unit vectors is
        # 2 - 2*cos). The reranker score orders the list but is not a calibrated percentage
        # on this OCR'd corpus, so it is returned separately for debugging only.
        source = {
            "pdf": meta.get("source", "unknown"),
            "page": meta.get("page", "?"),
            "distance": round(distance, 4),
            "relevance": round(min(1.0, max(0.0, 1.0 - distance / 2.0)), 4),
            "chunk_id": cand.chunk_id
        }
        if cand.rerank_score is not None:
            source["rerank_score"] = round(cand.rerank_score, 4)
        contexts.append(cand.text)
        sources.append(source)

    return contexts, sources


def build_prompt(question: str, contexts: list) -> str:
    context_text = "\n\n---\n\n".join(contexts)

    prompt = f"""
You are a legal assistant specializing in Kerala Government Gazettes and Orders.

Your responsibilities:
- Semantically understand the retrieved text
- Identify if it refers to a Government Order, Notification, Rule, or Authority
- Explain the **legal meaning and implication** in simple terms
- Combine multiple parts if the order spans sections

Rules:
1. Use ONLY the information present in the context
2. You MAY paraphrase, summarize, and interpret legally
3. Do NOT invent facts outside the context
4. If the context is insufficient, clearly say so

Language rules:
- If the question is in Malayalam → answer in Malayalam
- If the question is in English → answer in English
- If mixed → prefer Malayalam

IMPORTANT - Output Format:
- Use proper Markdown formatting
- Start with a clear heading (## for main topics)
- Use subheadings (###) for different sections
- Use bullet points (-) or numbered lists (1., 2., 3.) for multiple items
- Add blank lines between paragraphs
- Use **bold** for key terms and act names
- Keep paragraphs short (2-3 sentences max)
- Structure the answer logically with clear sections

Example structure:
## Main Topic/Act Name

Brief overview paragraph.

### Key Provisions
- Provision 1 explanation
- Provision 2 explanation

### Important Details
Short explanatory paragraph.

Context (extracted from official documents):
{context_text}

Question:
{question}

Answer (in well-structured Markdown):
"""
    return prompt.strip()


@app.route('/api/ask', methods=['POST'])
def ask():
    """Main endpoint for answering questions"""
    try:
        data = request.get_json(silent=True) or {}
        question = str(data.get('question') or '').strip()

        if not question:
            return jsonify({"error": "Question is required"}), 400

        # Retrieve relevant context
        contexts, sources = retrieve_context(question)

        if not contexts:
            return jsonify({
                "answer": "ബന്ധപ്പെട്ട രേഖകൾ കണ്ടെത്താനായില്ല.",
                "sources": []
            })

        # Build prompt and get the answer from Claude (the SDK retries 429/5xx/overloaded)
        prompt = build_prompt(question, contexts)

        try:
            answer = generate_answer(llm_client, prompt)
        except RefusalError as refusal:
            print(f"[LLM] {refusal}")
            return jsonify({
                "answer": "ഈ ചോദ്യത്തിന് ഉത്തരം നൽകാൻ കഴിയില്ല. / This question could not be answered.",
                "sources": sources
            })
        except anthropic.APIError as llm_error:
            return jsonify({
                "error": f"AI service temporarily unavailable: {str(llm_error)}",
                "sources": sources
            }), 503

        answer = answer or "ലഭ്യമായ രേഖകളിൽ നിന്ന് വ്യക്തമായ നിയമപരമായ വിശദീകരണം ലഭ്യമല്ല."

        return jsonify({
            "answer": answer,
            "sources": sources
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "Niyamadharshini API is running"})


# ---------- Retrieval evaluation viewer (/evals) ----------

EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval')
EVAL_RESULTS_DIR = os.path.join(EVAL_DIR, 'results')


def _read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _eval_runs():
    """Run directories under eval/results that contain a retrieval.json."""
    if not os.path.isdir(EVAL_RESULTS_DIR):
        return []
    runs = []
    for name in os.listdir(EVAL_RESULTS_DIR):
        path = os.path.join(EVAL_RESULTS_DIR, name, 'retrieval.json')
        if os.path.isfile(path):
            runs.append({
                "name": name,
                "modified": os.path.getmtime(path),
                "has_answers": os.path.isfile(os.path.join(EVAL_RESULTS_DIR, name, 'answers.json')),
            })
    return sorted(runs, key=lambda r: r["modified"], reverse=True)


@app.route('/evals')
def serve_evals():
    return send_from_directory('frontend', 'evals.html')


@app.route('/api/evals', methods=['GET'])
def list_evals():
    return jsonify({"runs": _eval_runs()})


@app.route('/api/evals/<run_name>', methods=['GET'])
def get_eval(run_name):
    # Only names returned by the listing are served (no path traversal)
    if run_name not in {r["name"] for r in _eval_runs()}:
        return jsonify({"error": "Unknown evaluation run"}), 404
    run_dir = os.path.join(EVAL_RESULTS_DIR, run_name)
    answers_path = os.path.join(run_dir, 'answers.json')
    golden_path = os.path.join(EVAL_DIR, 'golden_set.jsonl')
    with open(golden_path, encoding='utf-8') as f:
        golden = [json.loads(line) for line in f if line.strip()]
    return jsonify({
        "name": run_name,
        "retrieval": _read_json(os.path.join(run_dir, 'retrieval.json')),
        "answers": _read_json(answers_path) if os.path.isfile(answers_path) else None,
        "golden": golden,
    })


# Serve frontend
@app.route('/')
def serve_frontend():
    """Serve the main frontend page"""
    return send_from_directory('frontend', 'index.html')


@app.route('/frontend/<path:filename>')
def serve_frontend_files(filename):
    """Serve frontend static files"""
    return send_from_directory('frontend', filename)


# Serve PDF files from data folder
@app.route('/data/<path:filename>')
def serve_data_files(filename):
    """Serve PDF files from data folder"""
    return send_from_directory('data', filename)


if __name__ == '__main__':
    # 127.0.0.1 = this computer only; set FLASK_HOST=0.0.0.0 (with APP_PASSWORD) to share on a network
    host = os.getenv("FLASK_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("FLASK_PORT", "8000"))
    # Debug mode exposes Werkzeug's interactive debugger; only enable it on a trusted machine.
    debug = os.getenv("FLASK_DEBUG", "false").strip().lower() in ("1", "true", "yes")
    if host not in ("127.0.0.1", "localhost", "::1") and not APP_PASSWORD:
        print(f"[WARN] Listening on {host} without APP_PASSWORD: anyone on the network can use the app.")
    if debug and host not in ("127.0.0.1", "localhost", "::1"):
        print("[WARN] FLASK_DEBUG is on while listening on the network: the debugger allows code execution.")
    print("\n[SUCCESS] Niyamadharshini Server Starting...")
    print(f"[INFO] Open http://localhost:{port} in your browser")
    # use_reloader=False: the reloader would start a second process and load the models twice
    app.run(debug=debug, use_reloader=False, host=host, port=port)
