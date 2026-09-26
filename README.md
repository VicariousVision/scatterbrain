# Scatterbrain RAG

A clean, working document intelligence system using Retrieval-Augmented Generation (RAG). Upload PDF/TXT documents, and query them using natural language with LLM-powered responses grounded in your document content.

## Architecture

**Complete RAG Pipeline:**
1. **Document Upload** → Parse (PDF/TXT) → Chunk (LangChain) → Embed (Ollama) → Store (ChromaDB)
2. **Query** → Embed query → Search vector DB → Retrieve top-k chunks → Generate answer with LLM

**Tech Stack:**
- **Backend:** FastAPI, Python 3.11+
- **LLM & Embeddings:** Ollama (local inference)
- **Vector Store:** ChromaDB (persistent)
- **Document DB:** SQLite
- **Text Processing:** LangChain text splitters, pdfplumber
- **Frontend:** Streamlit

## Prerequisites

1. **Python 3.11+**
2. **Ollama** - Install from [ollama.ai](https://ollama.ai/)

## Setup

### 1. Install Ollama Models

```powershell
# Pull the LLM for generation (qwen3.5:0.8b is fast and lightweight)
ollama pull qwen3.5:0.8b

# Pull the embedding model
ollama pull nomic-embed-text

# Start Ollama server
ollama serve
```

### 2. Configure Environment

Copy `.env.example` to `.env`:

```powershell
cp .env.example .env
```

The defaults should work for local Ollama. Adjust if needed:

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:0.8b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_NUM_GPU=0

CHROMA_PERSIST_DIRECTORY=./backend/chroma_db
SQLITE_DB_PATH=./backend/documents.db

CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=5

BACKEND_URL=http://localhost:8000
```

### 3. Install Backend Dependencies

```powershell
cd backend
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies

```powershell
cd ../frontend
pip install -r requirements.txt
```

## Running the Application

### Start Backend (Terminal 1)

```powershell
cd backend
uvicorn main:app --reload
```

Backend will be available at `http://localhost:8000`

API docs at `http://localhost:8000/docs`

### Start Frontend (Terminal 2)

```powershell
cd frontend
streamlit run app.py
```

Frontend will open in your browser at `http://localhost:8501`

## Usage

### 1. Upload Documents

1. Navigate to the **Upload** page
2. Choose a PDF or TXT file
3. Click "Upload and Process"
4. Wait for status to change from `processing` → `completed`

The system will:
- Parse the document text
- Split it into semantic chunks (1000 chars with 200 overlap)
- Generate embeddings using Ollama
- Store chunks in ChromaDB vector database
- Save metadata in SQLite

### 2. Query Documents

1. Navigate to the **Chat** page
2. Ask questions about your uploaded documents
3. The system will:
   - Embed your query
   - Search for the 5 most relevant chunks
   - Generate a grounded answer using the LLM

## API Endpoints

### Documents

- `POST /documents/upload` - Upload PDF/TXT file
- `GET /documents/` - List all documents
- `GET /documents/{id}` - Get document status
- `DELETE /documents/{id}` - Delete document and its chunks

### Chat

- `POST /chat/query` - Query documents with RAG

### Health

- `GET /health` - Service health check

## Project Structure

```
scatterbrain/
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Settings (from .env)
│   ├── models/                    # Pydantic models
│   │   ├── chat.py
│   │   └── document.py
│   ├── routers/                   # API endpoints
│   │   ├── chat.py
│   │   ├── documents.py
│   │   └── health.py
│   ├── services/                  # Business logic
│   │   ├── chat_service.py        # RAG query orchestration
│   │   ├── document_service.py    # Upload & processing pipeline
│   │   ├── document_parser.py     # PDF/TXT parsing
│   │   ├── text_chunker.py        # LangChain text splitting
│   │   ├── vector_store.py        # ChromaDB wrapper
│   │   ├── document_db.py         # SQLite persistence
│   │   └── ollama_client.py       # Ollama API client
│   └── requirements.txt
├── frontend/
│   ├── app.py                     # Streamlit landing page
│   ├── api_client.py              # HTTP client for backend
│   ├── pages/
│   │   ├── 1_Upload.py            # Document upload UI
│   │   └── 2_Chat.py              # Chat interface
│   └── requirements.txt
├── .env                           # Local config (git-ignored)
├── .env.example                   # Config template
└── README.md
```

## Key Features

✅ **Clean RAG pipeline** - No graph complexity, pure vector-based retrieval  
✅ **Local-first** - Runs entirely on your machine with Ollama  
✅ **Persistent storage** - SQLite for metadata, ChromaDB for vectors  
✅ **Real-time status** - Upload polling with progress indicators  
✅ **Semantic chunking** - LangChain RecursiveCharacterTextSplitter  
✅ **Type-safe** - Pydantic models throughout  
✅ **Production-ready** - Proper error handling, logging, async/await  

## Configuration Options

All settings in `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3.5:0.8b` | LLM for answer generation |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_NUM_GPU` | `0` | GPU layers (0=CPU only) |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Number of chunks to retrieve |
| `CHROMA_PERSIST_DIRECTORY` | `./backend/chroma_db` | Vector DB path |
| `SQLITE_DB_PATH` | `./backend/documents.db` | Document DB path |

## Troubleshooting

**"Ollama is unavailable"**
- Make sure `ollama serve` is running
- Check `OLLAMA_BASE_URL` in `.env`
- Verify models are pulled: `ollama list` (should show qwen3.5:0.8b and nomic-embed-text)

**"No chunks produced from document"**
- Check if PDF is text-based (not scanned image)
- For TXT files, verify UTF-8 encoding

**Slow processing**
- Ollama CPU mode is slow; consider using GPU if available
- Reduce `CHUNK_SIZE` to process fewer chunks
- Smaller models (e.g., `phi3:mini`) are faster but less accurate

**Import errors**
- Ensure you're in the correct directory when running commands
- Reinstall dependencies: `pip install -r requirements.txt`

## Development

**Backend tests:**
```powershell
cd backend
pytest
```

**Backend with auto-reload:**
```powershell
uvicorn main:app --reload --log-level debug
```

**Clear vector database:**
```powershell
rm -rf backend/chroma_db
rm backend/documents.db
```

## License

MIT
