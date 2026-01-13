# Niyamadarshini

**Niyamadarshini** is an AI-powered legislative document intelligence system designed to make Kerala Legislative Assembly laws, rules, and proceedings **searchable, understandable, and accessible** to citizens, students, researchers, and judiciary staff.  
By combining **Retrieval-Augmented Generation (RAG)**, **multilingual embeddings**, and **adaptive retrieval strategies**, the system transforms complex legislative PDFs into a conversational knowledge platform.

---


## 💡 Solution Overview

Niyamadarshini converts legislative documents into structured, searchable knowledge units and enables users to ask questions in **Malayalam or English** to receive **clear, cited answers**.  
The system uses AI not as a knowledge source, but as a **reasoning engine grounded in official documents**.

---

## 🚀 Key Features

- 📄 **PDF Ingestion & Preprocessing** – Extracts and cleans text from searchable legislative PDFs  
- 🧠 **Multilingual Semantic Search** – Supports Malayalam and English using advanced embeddings  
- 🔎 **Adaptive RAG Pipeline** – Adjusts retrieval depth based on query complexity  
- 💬 **Conversational Chat Interface** – Natural-language Q&A with contextual memory  
- 📚 **Source-Cited Answers** – Every response includes document and page references  
- 🛠️ **Open-Source & Modular Architecture** – Pluggable models, databases, and deployment options  

---

## 🏗️ System Architecture
PDF Documents
↓
Text Extraction & Cleaning
↓
Chunking & Metadata Tagging
↓
Multilingual Embeddings (BGE-M3 / LaBSE)
↓
Vector Database (Chroma / FAISS)
↓
Adaptive Retrieval + Re-ranking
↓
LLM (Gemini / GPT / LLaMA via Ollama)
↓
Chat Interface with Citations



---

## 🧰 Tech Stack

### Backend
- Python 3.10+
- FastAPI
- LangChain
- SentenceTransformers / BGE-M3 / LaBSE
- ChromaDB / FAISS
- Ollama (optional local LLMs)

### Frontend
- React.js
- Tailwind CSS

### Models
- **Chat LLM**: Gemini / GPT / LLaMA / Mistral  
- **Embedding Models**: BGE-M3, LaBSE  

---


---

## ⚙️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/your-username/niyamadarshini.git
cd niyamadarshini

