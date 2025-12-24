"""
Multilingual RAG Chat Interface (Malayalam + English)
- ChromaDB (pre-ingested vectors)
- BGE-M3 embeddings
- Gemini (google.genai) for answer generation
"""

import os
from dotenv import load_dotenv
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from google import genai


# =========================
# 1. ENV + CONFIG
# =========================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found in .env")

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "malayalam_docs"
EMBED_MODEL_NAME = "BAAI/bge-m3"
TOP_K = 10


# =========================
# 2. INIT MODELS
# =========================

print("[INIT] Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

print("[INIT] Connecting to ChromaDB...")
chroma_client = PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)

print("[INIT] Initializing Gemini client...")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# =========================
# 3. RETRIEVAL
# =========================

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


# =========================
# 4. PROMPT (LEGAL-SAFE)
# =========================

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

Context (extracted from official documents):
{context_text}

Question:
{question}

Answer:
"""
    return prompt.strip()



# =========================
# 5. CHAT FUNCTION
# =========================

def ask(question: str):
    print("\n[SEARCH] Retrieving relevant documents...")
    contexts, sources = retrieve_context(question)

    if not contexts:
        print("❌ ബന്ധപ്പെട്ട രേഖകൾ കണ്ടെത്താനായില്ല.")
        return

    prompt = build_prompt(question, contexts)

    print("[LLM] Generating answer...")
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(response.text.strip() or "ലഭ്യമായ രേഖകളിൽ നിന്ന് വ്യക്തമായ നിയമപരമായ വിശദീകരണം ലഭ്യമല്ല.")

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)
    for s in sources:
        print(f"- {s['pdf']} | Page {s['page']} | distance={s['distance']}")


# =========================
# 6. CLI LOOP
# =========================

if __name__ == "__main__":
    print("\n🟢 Multilingual RAG Chat (type 'exit' to quit)\n")

    while True:
        query = input("❓ Ask: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        ask(query)
