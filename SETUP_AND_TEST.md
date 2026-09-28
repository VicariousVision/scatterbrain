# Setup and Test

## Install dependencies

From the repository root:

```powershell
cd backend
pip install -r requirements.txt
```

The backend supports two embedding providers. `sentence-transformers` is installed for the Hugging Face option; it downloads the selected model the first time it is used.

## Choose an embedding provider

### Ollama embeddings (default)

Start Ollama and pull the generation and embedding models:

```powershell
ollama serve
ollama pull qwen3.5:0.8b
ollama pull nomic-embed-text
```

Use these settings in `.env`:

```dotenv
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768
```

`EMBEDDING_DIMENSION` must match the selected Ollama embedding model.

### Hugging Face embeddings

Hugging Face models run locally through Sentence Transformers. Ollama is still required for answer generation, but no Ollama embedding model needs to be pulled.

```dotenv
EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_EMBEDDING_MODEL=litillabs/litil-embed-0.6b
HUGGINGFACE_DEVICE=cpu
HUGGINGFACE_NORMALIZE_EMBEDDINGS=true
```

`litillabs/litil-embed-0.6b` produces 1024-dimensional vectors. Scatterbrain detects the dimension automatically and applies its required query instruction to query embeddings; document chunks remain unprefixed. Set `HUGGINGFACE_DEVICE=cuda` when a compatible CUDA/PyTorch installation is available. The model is optimized for English, German, and Chinese legal retrieval.

## Important when changing models

SQLite-vector tables have a fixed dimension, and vectors from different models are not interchangeable. If you change `EMBEDDING_PROVIDER` or either embedding model, delete the database configured by `SQLITE_DB_PATH` and re-upload the documents:

```powershell
Remove-Item .\backend\scatterbrain.db -ErrorAction SilentlyContinue
```

Adjust the path if `SQLITE_DB_PATH` points somewhere else. On startup, Scatterbrain records the provider, model, and dimension and refuses to mix incompatible databases.

## Run the application

```powershell
cd backend
uvicorn main:app --reload
```

In another terminal:

```powershell
cd frontend
streamlit run app.py
```

Backend: http://localhost:8000
API docs: http://localhost:8000/docs
Frontend: http://localhost:8501

## Test the flow

1. Open the Upload page and upload a PDF or UTF-8 TXT file.
2. Wait for the document status to become `completed`.
3. Open Chat and ask a question answered by the document.
4. Ask a question unrelated to the document and verify the assistant says the context is insufficient.
5. Confirm the backend logs identify the selected provider and embedding dimension.

Run the backend test suite with:

```powershell
cd backend
pytest
```
