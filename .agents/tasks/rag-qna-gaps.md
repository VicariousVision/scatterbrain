# RAG Q&A gap analysis

## Summary answer
The end-to-end RAG Q&A chat is already built and wired. Nothing is missing to make it work. What is left is operational setup, a few quality gaps, and optional polish.

- Upload -> parse -> chunk -> embed -> store: implemented.
- Query -> embed -> vector search -> prompt -> LLM answer with sources: implemented. The frontend shows answers and citations.
- No TODO/FIXME/NotImplemented/stub markers in non-test Python code (grep).
- Tests exist for chat service, API citations, retrieval, vector store, document service and frontend citations (`backend/tests/`). I did not run them (read-only step).

Real gaps:
1. Conversation history is sent to the backend but never used in the prompt.
2. No explicit no-answer gate. With zero retrieved chunks the LLM is still called with "(No relevant documents found)".
3. Config and environment fragility: the relative DB path, and the embedding provider and DB must match.
4. No streaming. The 0.8b model on CPU is slow (180 s client timeout).

## Stage status

| Stage | Status | Where |
|---|---|---|
| Upload endpoint (PDF/TXT, 202 + background task) | implemented | `routers/documents.py`, `DocumentService.upload` / `_process_document` (per `.agents/tasks/processing-flow.md`) |
| Parse | implemented | `services/document_parser.py`, `text_cleaner.py` |
| Chunk | implemented | `legal_chunker.py` (parent/child) and `text_chunker.py` (generic fallback) |
| Embed | implemented | `embedding_provider.py` (ollama or huggingface), `OllamaClient.generate_embedding` |
| Store | implemented | `VectorStore.add_document` (atomic transaction, sqlite-vec `vec_chunks`) |
| Vector search | implemented | `VectorStore.search_candidates` / `search`; `RetrievalService.retrieve` / `search` (hybrid, diversified) |
| Prompt build | implemented | `chat_service.py`: `_SYSTEM_PROMPT`, `_USER_TEMPLATE`, `_select_within_budget` (context budget, injection-neutralized `<document>` blocks) |
| LLM call | implemented | `ChatService.query` -> `OllamaClient.chat(messages, think=...)` |
| Citations | implemented | `_citations`, `format_citation_label`; `models.content.Citation` |
| API | implemented | `POST /chat/query` in `routers/chat.py::chat_query`; 503 on `OllamaClientError`, 500 otherwise |
| Frontend chat | implemented | `frontend/pages/2_Chat.py` calls `api_client.chat_query`, renders answer, "Sources: ..." caption, retrieved-chunk count, 503 message |
| Frontend upload | implemented | `frontend/pages/1_Upload.py`, `api_client.upload_document` |

## Wiring check (`backend/main.py` lifespan)
`OllamaClient` -> health check (degraded start, not fatal) -> `create_embedding_provider` -> `VectorStore.initialize` -> `DocumentDB` (+ `fail_stale_processing`) -> `DocumentService`, `RetrievalService`, `ChatService(retrieval_service=...)` -> `documents_router.set_services`, `chat_router.set_services`, `search_router.set_services`. Routers are included. This matches the project convention (no `Depends`).

## Evidence for the gaps

1. History unused. `ChatRequest.history` is accepted and echoed back by `chat_query` (`updated_history`), and `2_Chat.py` sends `st.session_state["messages"]`. But `ChatService.query(user_query, top_k)` takes no history, and `messages` contains only the system prompt and one user turn. Follow-up questions ("what about section 2?") are retrieved and answered without context. The router also ignores the frontend's `citations` field in history, which is harmless.
2. No-answer handling is prompt-only. `ChatService.query` calls Ollama even when `kept` is empty. There is no similarity threshold in `RetrievalService`. A tiny 0.8b model may hallucinate despite the "do not guess" rule. Citations are derived from retrieved chunks, not from what the model actually cited, so sources can be listed for an answer that says "not found".
3. Config fragility (confirmed by existing notes `document-get-empty.md`, `processing-flow.md`, `missing-embeddings.md`):
   - `config.py` `sqlite_db_path` is cwd-relative. `.env` currently has an uncommitted change to `SQLITE_DB_PATH=backend\scatterbrain.db`, which resolves to `backend\backend\scatterbrain.db` if uvicorn starts from `backend/` (per the notes).
   - `.env.example` defaults to `EMBEDDING_PROVIDER=huggingface` (1024-dim) while the populated `backend/scatterbrain.db` was built with Ollama `nomic-embed-text` (768-dim). Pointing at the wrong DB or provider gives empty or wrong search. The notes show a live server once bound to `scatterbrain_huggingface.db`.
   - Both `OLLAMA_MODEL` (`qwen3.5:0.8b`) and, if using Ollama embeddings, `nomic-embed-text` must be pulled.
4. No streaming. `OllamaClient.chat` returns a full string and the UI shows a spinner. Not a blocker.
5. Minor: the chat page shows "Retrieved N chunks" using `len(result.source_texts)`, fine. CORS is `allow_origins=["*"]` with credentials, and there is no authentication on any endpoint. Acceptable for local-only use, but flag it if the backend is ever exposed.

## Remaining steps (priority order)
1. Run it: `ollama serve`; `ollama pull qwen3.5:0.8b`; pull `nomic-embed-text` if `EMBEDDING_PROVIDER=ollama`. Check `.env`: `EMBEDDING_PROVIDER`, `EMBEDDING_DIMENSION` and `SQLITE_DB_PATH` consistent with the DB you want. Start backend from `backend/`, frontend from `frontend/`, upload a document, wait for `completed`, then ask a question.
2. Fix the DB path. Either revert `.env` to `SQLITE_DB_PATH=scatterbrain.db` or resolve it against the backend directory in `config.py`. Log the resolved path and the embedding provider/model at startup.
3. Run `cd backend; pytest` to confirm the baseline is green.
4. Add a no-answer gate: in `ChatService.query`, when `kept` is empty (or best similarity is below a configurable threshold), return a fixed "I couldn't find this in your documents" answer with no citations and skip the LLM call. Add a unit test in `tests/test_chat_service.py`.
5. Use conversation history: pass the last N turns to `ChatService.query`. Optionally rewrite follow-ups into standalone queries before retrieval, and include recent turns in `messages`. Thread `request.history` through `routers/chat.py`.
6. Optional: streaming responses (`/chat/query` SSE plus `st.write_stream`), citation filtering to those the model actually referenced, a larger chat model for better answer quality, and the RAGAS evaluation (`pytest -m ragas`) for tuning.

## Conclusion
This is a working RAG Q&A app. The main risks are environment mismatches (DB path, provider/dimension) and answer quality on no-match or follow-up questions. Implementing steps 2, 4 and 5 closes the real functional gaps. I changed no files other than this report.
