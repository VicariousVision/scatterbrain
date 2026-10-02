# App Entry Point and Configuration

## `config.py`

Defines `Settings` (pydantic-settings `BaseSettings`) and a module-level `settings` instance imported everywhere. Values load from the repo-root `.env` (resolved relative to the file, so the working directory doesn't matter). Env var names are the field names in upper case, no prefix.

Used by: `main.py`, `DocumentDB`, `VectorStore`, `text_chunker`, `create_embedding_provider`, `evaluation/ragas_eval.py`.

### Parameters

Ollama

| Setting | Default | Purpose |
| --- | --- | --- |
| `ollama_base_url` | `http://localhost:11434` | Ollama server URL |
| `ollama_model` | `qwen3.5:0.8b` | Chat / generation model |
| `ollama_embedding_model` | `nomic-embed-text` | Embedding model when `embedding_provider=ollama` |
| `ollama_num_gpu` | `0` | GPU layers to offload (`0` = CPU only) |
| `ollama_think` | `False` | Chain-of-thought for thinking models; off keeps CPU answers fast |

Embeddings

| Setting | Default | Purpose |
| --- | --- | --- |
| `embedding_provider` | `huggingface` | `huggingface` (alias `hf`) or `ollama` |
| `embedding_dimension` | `768` | Vector size assumed for the Ollama provider (HF reads it from the model) |
| `huggingface_embedding_model` | `litillabs/litil-embed-0.6b` | Sentence Transformers model ID or path |
| `huggingface_device` | `cpu` | Torch device (`cpu`, `cuda`, ...) |
| `huggingface_normalize_embeddings` | `True` | L2-normalize vectors |
| `huggingface_query_prefix` | `Instruct: Retrieve text based on user query.\nQuery: ` | Instruction prepended to queries only |
| `huggingface_embedding_batch_size` | `32` | Texts per forward pass |

Storage, chunking, retrieval

| Setting | Default | Purpose |
| --- | --- | --- |
| `sqlite_db_path` | `scatterbrain.db` | Shared DB for document metadata and vectors |
| `chunk_size` | `500` | Max characters per chunk |
| `chunk_overlap` | `100` | Characters carried between prose chunks |
| `retrieval_top_k` | `5` | Default chunks returned by `VectorStore.search` |
| `backend_url` | `http://localhost:8000` | Backend URL (for clients) |

RAGAS evaluation (see [evaluation.md](evaluation.md))

| Setting | Default | Purpose |
| --- | --- | --- |
| `ragas_judge_model` | `""` | Judge model; empty falls back to `ollama_model` |
| `ragas_judge_max_tokens` | `2048` | Output cap per judge call |
| `ragas_max_samples` | `0` | Limit samples; `0` = all |
| `ragas_min_faithfulness` | `0.6` | Pass threshold |
| `ragas_min_answer_relevancy` | `0.6` | Pass threshold |
| `ragas_min_context_precision` | `0.6` | Pass threshold |
| `ragas_min_context_recall` | `0.6` | Pass threshold |

`OLLAMA_MAX_PARALLEL` is read directly from the environment by `ollama_client.py`, not through `Settings` (default `1`).

## `main.py`

Creates the FastAPI `app`, adds permissive CORS (`allow_origins=["*"]`), mounts the four routers, and runs the `lifespan` startup/shutdown sequence.

### Startup sequence

1. Build `OllamaClient` from `ollama_*` settings and `embedding_dimension`.
2. `health_check()` against Ollama. Failure is logged and the app starts in a degraded state.
3. `create_embedding_provider(settings, ollama_client)` picks HF or Ollama.
4. `VectorStore(embedding_provider).initialize()`: loads sqlite-vec, creates tables, validates embedding config. Failure aborts startup.
5. `DocumentDB()` then `fail_stale_processing()` to mark orphaned `processing` docs as `failed`.
6. `DocumentService(vector_store, document_db)` and `ChatService(vector_store, ollama_client, think=settings.ollama_think)`.
7. Register with routers: `documents.set_services(document_service)`, `chat.set_services(chat_service)`, `search.set_services(vector_store)`.

Shutdown: `vector_store.close()`.

### Security note

There is no authentication on any endpoint and CORS allows every origin. That's fine for local-only use; put auth in front before exposing the port on a network.
