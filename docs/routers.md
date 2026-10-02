# Routers (`backend/routers/`)

Routers are thin and retain module-level singleton injection from `main.py`; no FastAPI `Depends` service wiring is used.

| Router | Prefix | Injected service |
| --- | --- | --- |
| `health.py` | none | none |
| `documents.py` | `/documents` | `DocumentService` |
| `chat.py` | `/chat` | `ChatService` |
| `search.py` | `/search` | `RetrievalService` |

## Documents

- `POST /documents/upload` accepts `.pdf`/`.txt`, returns 202 `{document_id}`, and starts structured background ingestion.
- `GET /documents/` and `GET /documents/{id}` preserve existing status fields and add chunk count, source hash/title/version/parser, and `needs_reingestion`.
- `DELETE /documents/{id}` returns 204 after serialized atomic vector-store cleanup and metadata deletion; unknown IDs return 404.

Poll detail until `completed` or `failed`. Legacy completed rows may report `needs_reingestion=true` because old flat chunks cannot retroactively gain page/hierarchy provenance.

## Chat

`POST /chat/query` accepts the unchanged `{query, history}` body. `main.py` supplies validated `RETRIEVAL_TOP_K`; it is no longer hard-coded in the router.

The response retains:

```json
{"response": "...", "history": [], "retrieved_chunks": 5}
```

and adds default-empty `citations`, whose entries include filename, child/document IDs, section/clause, printed/PDF page range, revisions, heading, and display label. HTTP 503 represents local Ollama failure; other unexpected errors return 500. History remains display state and is not treated as source evidence.

## Search

`GET /search/?q=...&top_k=5&document_id=...` runs the same hybrid child ranking/diversification as chat but no LLM. `q` is required/non-empty, `top_k` is 1–50, and document restriction is optional.

Every old `SearchResultItem` field remains required/available. Structured score, hierarchy, pages, relationships, table/code/definition data, cross-references, extraction/parser versions, and short governing context are additive. `set_services(vector_store=...)` remains accepted as a compatibility adapter, while production injects the singleton `RetrievalService`.

## Health

`GET /health` returns liveness only; it does not guarantee Ollama/model readiness.
