# Niyamadharshini - Frontend & Backend Integration

## ✅ Setup Complete!

Your Malayalam RAG application is now fully integrated:
- **Frontend**: Modern, responsive UI with dark/light mode
- **Backend**: Flask API connected to ChromaDB + Gemini
- **Features**: Real-time Q&A with source citations

---

## 🚀 How to Run

### Step 1: Start the Backend API
```powershell
python api.py
```
This will start the server at `http://localhost:5000`

### Step 2: Open the Frontend
Open `frontend/index.html` in your browser, or run:
```powershell
Start-Process "frontend\index.html"
```

---

## 💡 Usage

1. Type your question in Malayalam or English in the text box
2. Click the send button (➤) or press Enter
3. Wait for the AI to search documents and generate an answer
4. View the answer and source documents with page numbers

---

## 🔧 Architecture

```
User Question (Frontend)
    ↓
Flask API (api.py)
    ↓
Embedding → ChromaDB → Retrieved Contexts
    ↓
Gemini 2.5 Flash → Answer Generation
    ↓
Response with Sources (Frontend)
```

---

## 📂 File Structure

```
d:\niyam\
├── api.py                 # Flask backend API
├── frontend/
│   └── index.html         # Frontend UI
├── scripts/
│   ├── chat.py           # Original terminal chat
│   ├── ingest_pdfs.py    # PDF ingestion pipeline
│   └── ...               # Other utilities
├── chroma_db/            # Vector database
└── .env                  # API keys
```

---

## ⚠️ Important Notes

- Backend must be running before using the frontend
- Make sure ChromaDB has ingested documents (run `ingest_pdfs.py` first if needed)
- GEMINI_API_KEY must be set in `.env` file

---

## 🎨 Features

### Frontend:
- ✨ Clean, modern UI with Malayalam support
- 🌓 Dark/Light mode toggle
- 📱 Responsive design
- ⌨️ Keyboard shortcuts (Enter to send)
- 📚 Source citations with page numbers

### Backend:
- 🔍 Semantic search with BGE-M3 embeddings
- 🤖 Gemini 2.5 Flash for answer generation
- 📊 Top 10 most relevant chunks
- 🌐 CORS enabled for local development

---

## 🐛 Troubleshooting

**Error: Connection refused**
- Make sure `api.py` is running
- Check that the API is on port 5000

**Error: No documents found**
- Run the ingestion pipeline first: `python scripts/ingest_pdfs.py`

**Error: GEMINI_API_KEY not found**
- Add your API key to `.env` file: `GEMINI_API_KEY=your_key_here`
