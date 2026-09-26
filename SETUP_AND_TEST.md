# Setup and Testing Guide

## Quick Start (After Prerequisites)

### 1. Install Dependencies

**Backend:**
```powershell
cd backend
pip install -r requirements.txt
```

**Frontend:**
```powershell
cd frontend
pip install -r requirements.txt
```

### 2. Start Ollama

Open a terminal and run:
```powershell
ollama serve
```

Keep this terminal running.

### 3. Pull Required Models

In another terminal:
```powershell
# LLM for generation (qwen3.5:0.8b is fast and lightweight)
ollama pull qwen3.5:0.8b

# Embedding model
ollama pull nomic-embed-text
```

### 4. Start Backend

```powershell
cd backend
uvicorn main:app --reload
```

You should see:
```
INFO:     Ollama connected at http://localhost:11434
INFO:     Vector store initialized
INFO:     Document database initialized
INFO:     Scatterbrain RAG backend started
INFO:     Application startup complete.
```

Backend API: http://localhost:8000
API Docs: http://localhost:8000/docs

### 5. Start Frontend

In a new terminal:
```powershell
cd frontend
streamlit run app.py
```

Browser will open at http://localhost:8501

## Testing the Complete Flow

### Test 1: Health Check

Visit http://localhost:8000/health

You should see:
```json
{
  "status": "healthy",
  "timestamp": "2024-..."
}
```

### Test 2: Upload a Document

1. Go to http://localhost:8501
2. Click **Upload** in sidebar
3. Upload a test PDF or create a test.txt file:
   ```
   This is a test document about machine learning.
   Machine learning is a subset of artificial intelligence.
   It involves training algorithms on data to make predictions.
   ```
4. Click "Upload and Process"
5. Wait for status to change to "completed"

**What happens behind the scenes:**
- Document is parsed
- Text is split into chunks (1000 chars, 200 overlap)
- Each chunk is embedded using Ollama
- Embeddings are stored in ChromaDB
- Metadata is saved in SQLite

### Test 3: Query the Document

1. Go to **Chat** page
2. Ask: "What is machine learning?"
3. System will:
   - Embed your query
   - Search ChromaDB for top 5 similar chunks
   - Send chunks + query to LLM
   - Return grounded answer

**Expected response:**
The assistant should answer based only on the document content, mentioning that machine learning is a subset of AI, involves training algorithms, etc.

### Test 4: Test Empty Context

Ask: "What is quantum computing?"

**Expected response:**
The assistant should say it doesn't have information about quantum computing in the uploaded documents (since we didn't upload anything about quantum computing).

## Verification Checklist

- [ ] Ollama is running and accessible at http://localhost:11434
- [ ] Models are pulled: `ollama list` shows qwen3.5:0.8b and nomic-embed-text
- [ ] Backend starts without errors
- [ ] Frontend loads in browser
- [ ] Can upload a document
- [ ] Document status changes to "completed"
- [ ] Can query and receive answers
- [ ] Answers are grounded in document content

## Common Issues

**ModuleNotFoundError during backend startup:**
```powershell
cd backend
pip install -r requirements.txt
```

**Ollama connection failed:**
- Check `ollama serve` is running
- Verify URL: http://localhost:11434
- Test manually: `curl http://localhost:11434`
- Verify models: `ollama list` (should show qwen3.5:0.8b and nomic-embed-text)

**ChromaDB errors:**
- Delete and recreate: `rm -rf backend/chroma_db`

**SQLite errors:**
- Delete and recreate: `rm backend/documents.db`

**Frontend can't connect to backend:**
- Verify backend is running: http://localhost:8000/health
- Check BACKEND_URL in .env: `http://localhost:8000`

## Architecture Verification

The refactored system has:
- ✅ No Neo4j code
- ✅ No graph extraction
- ✅ No entity/relationship code
- ✅ Clean RAG pipeline: parse → chunk → embed → store → retrieve → generate
- ✅ Persistent storage (SQLite + ChromaDB)
- ✅ Clean service separation
- ✅ Type-safe with Pydantic
- ✅ Async/await throughout

## Performance Notes

**CPU-only Ollama is slow:**
- First query takes ~10-30 seconds (model load)
- Subsequent queries: ~5-15 seconds
- Embedding generation: ~1-2 seconds per chunk

**To improve speed:**
- Use GPU if available (set OLLAMA_NUM_GPU > 0)
- Use smaller models (phi3:mini)
- Reduce chunk size to process fewer chunks
- Reduce RETRIEVAL_TOP_K to fetch fewer chunks
