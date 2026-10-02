# App Entry Point and Configuration

## `config.py`

`Settings` uses pydantic-settings and loads the repo-root `.env` regardless of process working directory. Environment names are upper-case field names with no prefix.

### Core settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `ollama_base_url` | `http://localhost:11434` | Local Ollama URL |
| `ollama_model` | `qwen3.5:0.8b` | Generation model |
| `ollama_embedding_model` | `nomic-embed-text` | Ollama embedding model |
| `ollama_num_gpu` | `0` | GPU layers (`0` = CPU) |
| `ollama_num_ctx` | `8192` | Ollama context and RAG budget basis |
| `ollama_think` | `False` | Thinking-mode switch |
| `embedding_provider` | `huggingface` | `huggingface`/`hf` or `ollama` |
| `embedding_dimension` | `768` | Expected Ollama vector dimension |
| `huggingface_embedding_model` | `litillabs/litil-embed-0.6b` | Sentence Transformers model |
| `huggingface_device` | `cpu` | Torch device |
| `huggingface_normalize_embeddings` | `True` | Normalize HF vectors |
| `huggingface_query_prefix` | model instruction | Query-only prefix |
| `huggingface_embedding_batch_size` | `32` | HF batch size |
| `sqlite_db_path` | `scatterbrain.db` | Shared metadata/vector database |
| `backend_url` | `http://localhost:8000` | Client backend URL |

### Generic and legal chunking

| Setting | Default | Purpose |
| --- | ---: | --- |
| `chunk_size` | `1000` | Generic/final recursive fallback only |
| `chunk_overlap` | `200` | Generic fallback overlap only |
| `legal_chunk_target_chars` | `900` | Searchable legal child target |
| `legal_chunk_min_chars` | `450` | Soft intact minimum |
| `legal_chunk_hard_max_chars` | `1400` | Strict searchable maximum |
| `legal_forced_split_overlap_chars` | `120` | Tail only for a forced continuous provision |
| `legal_parent_max_chars` | `8000` | Intermediate-parent threshold |

### Retrieval

| Setting | Default | Purpose |
| --- | ---: | --- |
| `retrieval_candidate_pool` | `10` | Vector children before hybrid ranking |
| `retrieval_top_k` | `5` | Final diversified primary children |
| `retrieval_max_children_per_parent` | `2` | First-pass parent cap |
| `retrieval_max_children_per_section` | `3` | First-pass section cap |

Validation requires generic overlap < size, legal minimum ≤ target ≤ hard maximum, forced overlap below target/hard maximum, parent maximum ≥ child hard maximum, candidate pool ≥ final top-k, and positive limits.

### RAGAS settings

`ragas_judge_model` (empty → chat model), `ragas_judge_max_tokens=2048`, `ragas_max_samples=0`, and `ragas_min_faithfulness`, `ragas_min_answer_relevancy`, `ragas_min_context_precision`, `ragas_min_context_recall` (all `0.6`). See [evaluation.md](evaluation.md).

`OLLAMA_MAX_PARALLEL` is read directly by `ollama_client.py` (default `1`).

## `main.py`

The FastAPI lifespan uses the required module-level router injection convention:

1. Build `OllamaClient` from validated settings and perform a non-fatal health check.
2. Select the embedding provider.
3. Initialize `VectorStore`; schema/provider failures abort startup without deleting data.
4. Initialize/migrate `DocumentDB` and fail stale `processing` records.
5. Create `DocumentService`, one `RetrievalService`, and `ChatService(retrieval_service=...)`.
6. Register services through `documents.set_services`, `chat.set_services`, and `search.set_services`.
7. Close the vector connection at shutdown.

CORS currently allows every origin and endpoints have no authentication. Keep the service local or add an authenticated reverse proxy before network exposure.
