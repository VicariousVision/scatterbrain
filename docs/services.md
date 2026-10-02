# Services (`backend/services/`)

Business logic. Everything I/O-bound is `async`; CPU-heavy sync work (PDF parsing, chunking, HF encoding) is pushed to worker threads with `asyncio.to_thread`.

| Service | Role | Used by | Uses |
| --- | --- | --- | --- |
| `DocumentService` | Upload + background ingestion pipeline | documents router | `DocumentDB`, `document_parser`, `text_cleaner`, `text_chunker`, `VectorStore` |
| `DocumentDB` | Document metadata (sync `sqlite3`) | `DocumentService`, `main.py` | `settings`, `DocumentRecord` |
| `document_parser` | PDF/TXT → text | `DocumentService`, evaluation | `pdfplumber` |
| `text_cleaner` | Normalize parsed text | `DocumentService`, evaluation | stdlib only |
| `text_chunker` | Text → chunks | `DocumentService`, evaluation | `settings` |
| `EmbeddingProvider` | Text → vectors | `VectorStore`, evaluation | Sentence Transformers or `OllamaClient` |
| `VectorStore` | Store/search embeddings (aiosqlite + sqlite-vec) | `DocumentService`, `ChatService`, search router | `EmbeddingProvider`, `settings` |
| `ChatService` | RAG: retrieve → prompt → generate | chat router, evaluation | `VectorStore`, `OllamaClient` |
| `OllamaClient` | Async HTTP client for Ollama | `ChatService`, embedding provider (ollama mode), evaluation | `httpx` |

---

## `document_service.py` — `DocumentService`

Orchestrates uploads and runs ingestion as a fire-and-forget `asyncio` task.

Constructor

| Param | Type | Notes |
| --- | --- | --- |
| `vector_store` | `VectorStore` | Where chunks/embeddings go |
| `document_db` | `DocumentDB` | Persistent status records |

Methods

| Method | Params | Returns | Notes |
| --- | --- | --- | --- |
| `upload` (async) | `filename: str`, `content: bytes` | `DocumentRecord` (`processing`) | Creates UUID record, schedules `_process_document` |
| `list_documents` | none | `List[DocumentRecord]` | Delegates to `DocumentDB.list_all` |
| `get_document` | `document_id: str` | `DocumentRecord \| None` | |
| `delete_document` (async) | `document_id: str` | `None` | Deletes vectors, then metadata row |

Pipeline (`_process_document`)
1. `parse_document` (thread) → 2. `clean_text` → 3. `chunk_text` (thread) → 4. under a `Semaphore(1)`: `VectorStore.delete_by_filename(filename)` then `add_document` → 5. status `completed` with `chunk_count`.

Any exception marks the document `failed` with the error message. Re-uploading a filename replaces the old chunks in the vector store, but the old document's metadata row stays.

---

## `document_db.py` — `DocumentDB`

Synchronous `sqlite3` store for the `documents` table in the same DB file as the vectors (10 s busy timeout to coexist with the aiosqlite connection). A new connection is opened per call.

Constructor: `db_path: str | None = None` (defaults to `settings.sqlite_db_path`). Creates the schema on init.

Schema: `documents(document_id PK, filename, uploaded_at, status, error, chunk_count)`.

| Method | Params | Returns |
| --- | --- | --- |
| `create` | `record: DocumentRecord` | `None` |
| `get` | `document_id: str` | `DocumentRecord \| None` |
| `list_all` | none | `List[DocumentRecord]`, newest first |
| `update_status` | `document_id`, `status`, `error=None`, `chunk_count=None` | `None` |
| `fail_stale_processing` | `error: str` (default message) | `int` rows reclaimed |
| `delete` | `document_id: str` | `None` |

---

## `document_parser.py`

`parse_document(filename: str, content: bytes) -> str`

- Dispatches on extension: `.pdf` → pdfplumber, `.txt` → UTF-8 decode.
- PDFs: tables are detected per page and rendered as Markdown tables; prose outside table boxes is extracted separately so cells aren't duplicated. Pages are joined with blank lines.
- Raises `ValueError` for unsupported extensions, `DocumentParsingError` for corrupt/undecodable files.

---

## `text_cleaner.py`

`clean_text(text: str) -> str`

Conservative normalization before chunking: NFKC, `\r\n` → `\n`, rejoin `hyph-\nenated` words, strip control and zero-width chars, collapse in-line whitespace while keeping leading indentation, cap blank lines at one. Does not lowercase or remove punctuation/stopwords. Markdown table rows keep their pipes. Returns `""` for blank input.

---

## `text_chunker.py`

`chunk_text(text: str) -> List[str]`

Reads `settings.chunk_size` and `settings.chunk_overlap` at call time.

- Splits the text into prose and Markdown-table segments.
- Prose: custom recursive splitter over separators `["\n\n", "\n", ". ", " ", ""]` with character overlap.
- Tables: split on row boundaries with the header + separator row repeated in every chunk. A single row longer than `chunk_size` becomes its own chunk. No overlap for tables.

Note: this is a hand-rolled splitter modeled on LangChain's `RecursiveCharacterTextSplitter`; it doesn't import LangChain.

---

## `embedding_provider.py`

### `EmbeddingProvider` (Protocol)
What `VectorStore` needs from an embedder:

| Member | Type |
| --- | --- |
| `provider_name` | `str` |
| `model_name` | `str` |
| `embedding_dimension` | `int` |
| `generate_embedding(text)` | async → `List[float]` |
| `generate_embeddings(texts)` | async → `List[List[float]]` (optional in practice) |

Optional: `generate_query_embedding(text)`, used by `VectorStore` for queries when present.

### `create_embedding_provider(settings, ollama_client)`
Returns `ollama_client` when `embedding_provider == "ollama"`, a `HuggingFaceEmbeddingProvider` for `huggingface`/`hf`, otherwise raises `ValueError`.

### `HuggingFaceEmbeddingProvider`

| Param | Type | Default | Notes |
| --- | --- | --- | --- |
| `model_name` | `str` | required | HF ID or local path |
| `device` | `str` | `"cpu"` | Torch device |
| `normalize_embeddings` | `bool` | `True` | L2-normalize |
| `query_prefix` | `str` | `""` | Prepended in `generate_query_embedding` only |
| `batch_size` | `int` | `32` | Min 1 |

Loads the model synchronously in `__init__` and reads `embedding_dimension` from it. Encoding runs in a worker thread. Requires `sentence-transformers`.

---

## `vector_store.py` — `VectorStore`

aiosqlite connection with the sqlite-vec extension loaded. One long-lived connection for the app lifetime.

Constructor: `embedding_provider: EmbeddingProvider` (dimension must be > 0).

Schema
- `document_chunks(id, chunk_id UNIQUE, document_id, filename, chunk_index, content, created_at)` + index on `document_id`
- `vec_chunks` virtual table: `vec0(chunk_id TEXT PRIMARY KEY, embedding FLOAT[dim])`
- `embedding_metadata(provider, model, dimension)` single row

On `initialize()` the stored provider/model/dimension must match the configured provider or startup fails. Changing embedding model means deleting the DB and re-ingesting.

| Method | Params | Returns | Notes |
| --- | --- | --- | --- |
| `initialize` (async) | none | `None` | Connect, load extension, create schema, validate |
| `close` (async) | none | `None` | |
| `add_document` (async) | `document_id`, `filename`, `chunks: List[str]` | `int` chunks stored | Batch-embeds when provider supports it; `chunk_id = {document_id}_chunk_{i}` |
| `search` (async) | `query`, `top_k=None` (→ `settings.retrieval_top_k`), `document_id=None` | `List[dict]` | Embeds query (with prefix if available), brute-force `vec_distance_cosine`, ascending distance |
| `delete_document` (async) | `document_id` | `None` | Removes rows from both tables |
| `delete_by_filename` (async) | `filename` | `None` | Same, keyed by filename |

Search result dict:
```python
{"id": str, "text": str,
 "metadata": {"document_id": str, "filename": str, "chunk_index": int},
 "similarity": float}  # 1 - cosine distance
```

---

## `chat_service.py` — `ChatService`

| Param | Type | Default | Notes |
| --- | --- | --- | --- |
| `vector_store` | `VectorStore` | required | Retrieval |
| `ollama_client` | `OllamaClient` | required | Generation |
| `think` | `bool \| None` | `None` | Passed to Ollama `think`; `None` = model default |

`query(user_query: str, top_k: int = 5) -> Tuple[str, List[str]]`

Searches the vector store, joins chunk texts with blank lines (or `"(No relevant documents found)"`), fills a prompt that tells the model to answer only from the context, and calls `OllamaClient.generate`. Returns `(answer, chunk_texts)`. `OllamaClientError` propagates to the router (→ 503).

---

## `ollama_client.py` — `OllamaClient`

Async `httpx` wrapper around Ollama's REST API. Also satisfies `EmbeddingProvider` (single-text only, no batching, no query prefix).

Constructor

| Param | Type | Default | Notes |
| --- | --- | --- | --- |
| `base_url` | `str` | required | Trailing `/` stripped |
| `model` | `str` | required | Generation model |
| `embedding_model` | `str \| None` | `None` → `model` | |
| `num_gpu` | `int` | `0` | Sent as `options.num_gpu` |
| `embedding_dimension` | `int` | `768` | Must be > 0 |

Attributes set for the provider protocol: `provider_name="ollama"`, `model_name=embedding_model`, `embedding_dimension`.

| Method | Params | Returns | Endpoint |
| --- | --- | --- | --- |
| `generate_embedding` | `text` | `list[float]` | `POST /api/embeddings` |
| `generate` | `prompt`, `json_schema=None`, `max_tokens=None`, `think=None`, `temperature=None` | `str` | `POST /api/generate` (non-streaming) |
| `health_check` | none | `bool` | `GET /api/tags` |

Reliability behaviour
- All calls share a per-event-loop `asyncio.Semaphore(OLLAMA_MAX_PARALLEL)` (env var, default `1`), so embeddings and generation never overlap.
- Timeouts: connect/write/pool 30 s, read 1000 s; health check 5 s.
- `ConnectTimeout` / `ConnectError` retried up to 3 attempts with 2 s, 4 s back-off. HTTP status and response-format errors are not retried.
- Failures raise `OllamaClientError`.
