"""
Flask API for Niyamadharshini RAG Chat
Wraps the chat.py functionality for frontend integration
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import time
from dotenv import load_dotenv
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from google import genai

# Load environment
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found in .env")

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "malayalam_docs"
EMBED_MODEL_NAME = "BAAI/bge-m3"
TOP_K = 10

# Initialize models (load once at startup)
print("[INIT] Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

print("[INIT] Connecting to ChromaDB...")
chroma_client = PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)

print("[INIT] Initializing Gemini client...")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def retrieve_context(query: str, top_k: int = TOP_K):
    """Embed query and retrieve relevant chunks from ChromaDB."""
    query_embedding = embed_model.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    contexts = []
    sources = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        contexts.append(doc)
        sources.append({
            "pdf": meta.get("source", "unknown"),
            "page": meta.get("page", "?"),
            "distance": round(dist, 4)
        })

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


def call_gemini_with_retry(prompt: str, retries: int = 5, initial_delay: int = 2) -> str:
    """
    Call Gemini API with simple exponential-backoff retry logic.
    We avoid importing google.api_core here; instead we treat
    typical transient errors (503/429/unavailable/rate limit) as retryable
    based on the exception message.
    """

    def is_retryable(e: Exception) -> bool:
        msg = str(e).lower()
        return (
            "503" in msg
            or "429" in msg
            or "unavailable" in msg
            or "resource exhausted" in msg
            or "rate" in msg
        )

    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return (response.text or "").strip()
        except Exception as e:
            last_error = e

            # Retry only for transient errors and if we still have attempts left
            if is_retryable(e) and attempt < retries - 1:
                wait_time = initial_delay * (2**attempt)  # 2,4,8,...
                print(
                    f"[RETRY] Attempt {attempt + 1}/{retries} failed ({e}). "
                    f"Waiting {wait_time}s before retry..."
                )
                time.sleep(wait_time)
                continue

            # Non‑retryable or last attempt: re-raise
            raise

    # Should not normally reach here, but keep a clear error message
    raise Exception(
        f"Gemini API unavailable after {retries} retries. Please try again later."
    ) from last_error


@app.route('/api/ask', methods=['POST'])
def ask():
    """Main endpoint for answering questions"""
    try:
        data = request.json
        question = data.get('question', '').strip()

        if not question:
            return jsonify({"error": "Question is required"}), 400

        # Retrieve relevant context
        contexts, sources = retrieve_context(question)

        if not contexts:
            return jsonify({
                "answer": "ബന്ധപ്പെട്ട രേഖകൾ കണ്ടെത്താനായില്ല.",
                "sources": []
            })

        # Build prompt and get answer from Gemini with retry logic
        prompt = build_prompt(question, contexts)
        
        try:
            answer = call_gemini_with_retry(prompt)
        except Exception as gemini_error:
            return jsonify({
                "error": f"AI service temporarily unavailable: {str(gemini_error)}",
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
    print("\n🟢 Niyamadharshini Server Starting...")
    print("💡 Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=8000)
