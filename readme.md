# Niyamadarshini

**Niyamadarshini** is an AI-powered legislative document intelligence system designed to make Kerala Legislative Assembly laws, rules, and proceedings **searchable, understandable, and accessible** to citizens, students, researchers, and judiciary staff.  
By combining **Retrieval-Augmented Generation (RAG)**, **multilingual embeddings**, and **adaptive retrieval strategies**, the system transforms complex legislative PDFs into a conversational knowledge platform.

---


## 💡 Solution Overview

Niyamadarshini converts legislative documents into structured, searchable knowledge units and enables users to ask questions in **Malayalam and English** to receive **clear, cited answers**.  
The system uses AI not as a knowledge source, but as a **reasoning engine grounded in official documents**.

---
<img width="1918" height="997" alt="Screenshot 2026-01-20 132046" src="https://github.com/user-attachments/assets/2d09f301-b5b5-463a-a7c7-99716f77eb22" />
<img width="1914" height="813" alt="Screenshot 2026-01-20 132112" src="https://github.com/user-attachments/assets/6b11025c-f26f-4b8d-91ee-b018aa526066" />



## How It Works

```text
PDF files
  -> PyMuPDF text extraction
  -> PaddleOCR/Tesseract fallback for scanned pages
  -> Malayalam-safe text cleaning
  -> LangChain text chunking
  -> BAAI/bge-m3 embeddings
  -> ChromaDB persistent vector store

User question
  -> BAAI/bge-m3 query embedding
  -> Top 10 ChromaDB matches
  -> Gemini 2.5 Flash answer generation
  -> Answer and source pages in the browser
```



## Current Technology Stack

### Backend

- Python 3.10+
- Flask web server
- Flask-CORS
- Google Gemini API (`gemini-2.5-flash`)
- SentenceTransformers
- `BAAI/bge-m3` multilingual embedding model
- ChromaDB persistent vector database
- `python-dotenv` for environment configuration

### Frontend

- Plain HTML, CSS, and vanilla JavaScript
- Marked.js from jsDelivr for Markdown rendering
- Browser `localStorage` for chat history
- Flask serves the frontend; no separate frontend server is required

### PDF Processing

- PyMuPDF (`fitz`) for normal PDF text extraction
- PaddleOCR for scanned-page OCR
- Tesseract OCR with Malayalam and English support as a fallback
- LangChain `RecursiveCharacterTextSplitter` for chunking

## Architecture
<img width="1596" height="2130" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/08046493-9953-4fef-a466-32c4b129d0f3" />

## Project Structure

```text
d:\niyam\
├── api.py                    # Flask API and frontend server
├── frontend/
│   └── index.html            # Chat interface, CSS, and JavaScript
├── scripts/
│   ├── ingest_pdfs.py        # Complete PDF ingestion pipeline
│   ├── ocr_extract.py        # PDF extraction and OCR fallback
│   ├── clean_text.py         # Text normalization and cleanup
│   ├── chunk_text.py         # Text chunking and metadata creation
│   ├── embed_store.py        # Embedding generation and ChromaDB storage
│   ├── chat.py               # Terminal chat client
│   └── test.py               # Gemini API test
├── data/                     # Source PDF files
├── chroma_db/                # Persistent ChromaDB data
├── rag_env/                  # Python virtual environment
└── INTEGRATION_GUIDE.md      # Short integration notes
```

## Setup

Use the existing 64-bit virtual environment from PowerShell:

```powershell
cd D:\niyam
D:\niyam\rag_env\Scripts\Activate.ps1
```

The application requires a Gemini API key. Store it in a `.env` file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key
```

Never commit `.env` or expose the API key in source code. If a key has been committed or shared, revoke it and create a new one.

## Ingest Documents

Place PDF files in `data\`, then run:

```powershell
D:\niyam\rag_env\Scripts\python.exe scripts\ingest_pdfs.py
```

The default ingestion settings are:

- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Minimum chunk size: 200 characters
- Maximum chunks: 40,000
- ChromaDB path: `chroma_db`
- Collection: `malayalam_docs`

The pipeline skips PDFs that are already present in the collection. It extracts text, applies OCR when necessary, cleans the text, creates chunks with page metadata, generates BGE-M3 vectors, and stores them in ChromaDB.

## Run The Application

From `D:\niyam`, start the backend and frontend together:

```powershell
D:\niyam\rag_env\Scripts\python.exe api.py
```

Open the application at:

```text
http://localhost:5000
```

The Flask server serves `frontend\index.html` at `/`, the API at `/api/*`, and source PDFs from `/data/*`.

The server loads the embedding model before it starts listening. The first startup can take time and requires several gigabytes of available memory. Flask debug reloading is disabled to prevent the large embedding model from being loaded twice.

## API Endpoints

### Health check

```text
GET /api/health
```

Example response:

```json
{
  "status": "ok",
  "message": "Niyamadharshini API is running"
}
```

### Ask a question

```text
POST /api/ask
Content-Type: application/json
```

Request:

```json
{
  "question": "What does this government order say?"
}
```

Response:

```json
{
  "answer": "## Government Order\n\n...",
  "sources": [
    {
      "pdf": "gazette.pdf",
      "page": "4",
      "distance": 0.231
    }
  ]
}
```

## Terminal Chat

The original command-line client can be run with:

```powershell
D:\niyam\rag_env\Scripts\python.exe scripts\chat.py
```

Type a Malayalam or English question. Type `exit` or `quit` to stop.

## Frontend Features

- Malayalam and English questions
- Markdown-formatted answers
- Source links with PDF names and page numbers
- Dark/light theme toggle
- New chat, chat loading, and chat deletion
- Browser-based chat history
- Copy-answer button
- Casual greetings handled locally without an API request

Chat history is stored in the browser's `localStorage`; it is not stored in ChromaDB or on the Flask server.

## Troubleshooting

### `memory allocation ... failed`

The `BAAI/bge-m3` model is large. Close memory-heavy applications and ensure Windows has sufficient pagefile space. The server now runs with Flask debug reloading disabled so the model is not duplicated.

### Connection refused

Confirm that `api.py` is still running and open:

```text
http://localhost:5000/api/health
```

### Collection not found

Run the ingestion pipeline first:

```powershell
D:\niyam\rag_env\Scripts\python.exe scripts\ingest_pdfs.py
```

The backend expects the ChromaDB collection `malayalam_docs` to exist.

### No documents found

Confirm that PDF files are directly inside `data\`, not inside a nested folder, and run ingestion again.

### Gemini errors or quota errors

Check `GEMINI_API_KEY` in `.env`, confirm that the key is active, and retry later if Gemini returns a rate-limit or service-unavailable response.


