# Models (`backend/models/`)

Pydantic models for API I/O and internal data transfer. No ORM or database logic lives here.

## `models/chat.py`

### `ChatRequest`
Body of `POST /chat/query`. Used by `routers/chat.py`.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `query` | `str` | required | User question |
| `history` | `list[dict]` | `[]` | Prior messages `{"role": "user" \| "assistant", "content": str}` |

### `ChatResponse`
Response of `POST /chat/query`.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `response` | `str` | required | LLM answer |
| `history` | `list[dict]` | required | Input history plus this user/assistant turn |
| `retrieved_chunks` | `int` | `0` | Number of chunks used as context (count, not text) |

## `models/document.py`

### `DocumentRecord`
Internal tracking record. Created by `DocumentService.upload`, persisted and loaded by `DocumentDB`.

| Field | Type | Notes |
| --- | --- | --- |
| `document_id` | `str` | UUID4 |
| `filename` | `str` | Original upload name |
| `uploaded_at` | `datetime` | UTC |
| `status` | `Literal["processing", "completed", "failed"]` | `processing` → `completed` \| `failed` |
| `error` | `str \| None` | Failure reason |

The `documents` table also stores `chunk_count`, but it isn't a field on this model, so it's not returned by the API.

### `UploadResponse`
Response of `POST /documents/upload` (202).

| Field | Type |
| --- | --- |
| `document_id` | `str` |

### `DocumentListItem`
Element of `GET /documents` and response of `GET /documents/{id}`. Same fields as `DocumentRecord`, with `status` typed as plain `str`.

### `SearchResultItem`
Element of `GET /search`. Built in `routers/search.py` from `VectorStore.search` result dicts.

| Field | Type | Source in search result |
| --- | --- | --- |
| `chunk_id` | `str` | `id` (`{document_id}_chunk_{i}`) |
| `document_id` | `str` | `metadata.document_id` |
| `filename` | `str` | `metadata.filename` |
| `chunk_index` | `int` | `metadata.chunk_index` |
| `text` | `str` | `text` |
| `similarity` | `float` | `similarity` (`1 - cosine distance`) |
