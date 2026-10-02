# Routers (`backend/routers/`)

Thin HTTP layer. Each router holds a module-level service singleton set by `main.py` via `set_services(...)`; calling an endpoint before registration raises `RuntimeError`.

| Router | Prefix | Depends on |
| --- | --- | --- |
| `health.py` | none | nothing |
| `documents.py` | `/documents` | `DocumentService` |
| `chat.py` | `/chat` | `ChatService` |
| `search.py` | `/search` | `VectorStore` |

## `health.py`

`GET /health` → `200 {"status": "ok"}`. Liveness only; doesn't check Ollama or the DB.

## `documents.py`

`set_services(document_service: DocumentService)`

| Method & path | Input | Success | Errors | Calls |
| --- | --- | --- | --- | --- |
| `POST /documents/upload` | multipart `file` (`.pdf` / `.txt`) | `202` `UploadResponse` | `400` wrong extension | `DocumentService.upload` |
| `GET /documents/` | none | `200` `List[DocumentListItem]`, newest first | | `DocumentService.list_documents` |
| `GET /documents/{document_id}` | path `document_id` | `200` `DocumentListItem` | `404` | `DocumentService.get_document` |
| `DELETE /documents/{document_id}` | path `document_id` | `204` | `404` | `get_document`, then `delete_document` |

Upload returns immediately; poll `GET /documents/{id}` until status is `completed` or `failed`. The whole file is read into memory and there is no size limit.

## `chat.py`

`set_services(chat_service: ChatService)`

| Method & path | Input | Success | Errors | Calls |
| --- | --- | --- | --- | --- |
| `POST /chat/query` | `ChatRequest` | `200` `ChatResponse` | `503` on `OllamaClientError`, `500` otherwise | `ChatService.query(user_query, top_k=5)` |

Behaviour worth knowing:
- `top_k` is hard-coded to `5`, not read from `settings.retrieval_top_k`.
- `history` is echoed back with the new turn appended but is not sent to the LLM, so each query is answered independently.

## `search.py`

`set_services(vector_store: VectorStore)`

`GET /search/` returns `List[SearchResultItem]` ordered by similarity (highest first). No LLM call.

| Query param | Type | Default | Constraint |
| --- | --- | --- | --- |
| `q` | `str` | required | `min_length=1` |
| `top_k` | `int` | `5` | `1`–`50` |
| `document_id` | `str \| None` | `None` | Restrict to one document |

Calls `VectorStore.search(query=q, top_k=top_k, document_id=document_id)`.
