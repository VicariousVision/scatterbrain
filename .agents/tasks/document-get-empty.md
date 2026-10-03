# GET /documents/dfa8e4cc-... returns nothing: investigation

## Summary
The handler is fine. The running server answers 404 `Document '...' not found.` because it is reading a different SQLite file than the one that holds the row.

- The row exists only in `backend\scatterbrain.db` (status `completed`, "Currency and Exchanges Manual for Authorised Dealers.pdf").
- The live server at 127.0.0.1:8000 lists documents that exist only in `backend\scatterbrain_huggingface.db`. That file has 5 rows, and `dfa8e4cc-...` is not one of them.
- The reported "nothing" is most likely the 404 JSON error body, or the browser/client hiding it.

## Evidence
1. Handler, `backend\routers\documents.py::get_document`:
   - It calls `svc.get_document(id)`, which calls `DocumentDB.get` (`document_service.py:66`, `document_db.py::get`).
   - On `None` it raises `HTTPException(404, "Document '<id>' not found.")`.
   - It never returns an empty body or null.
   - `response_model=DocumentListItem`.
2. Live request (read-only GET): `/documents/dfa8e4cc-...` gave HTTP 404. `/documents/` returned the ids `d4b71990-...` and others, all from `scatterbrain_huggingface.db`.
3. Read-only sqlite queries:
   - `backend\scatterbrain.db`: 1 document, exactly `dfa8e4cc-...`, completed.
   - `backend\scatterbrain_huggingface.db`: 5 documents (`03f71b0a`, `ec5f1c1a`, `b4970ef8`, `644dd17d`, `d4b71990`), none matching.
   - `.agents\backups\scatterbrain-20261002-201152.db`: 0 documents.
   - No repo-root `scatterbrain.db` exists.
4. DB path selection:
   - `backend\config.py`: `sqlite_db_path = "scatterbrain.db"`, a relative path, so it depends on the process cwd.
   - `.env` has `SQLITE_DB_PATH=scatterbrain.db`.
   - `DocumentDB()` and `VectorStore()` both use `settings.sqlite_db_path` (`main.py:88,99`).
   - Nothing in the source references `scatterbrain_huggingface.db`. The live server must therefore have been started with a `SQLITE_DB_PATH=scatterbrain_huggingface.db` environment variable. Environment variables override `.env` in pydantic-settings. I could not confirm this directly from the process environment.
5. The id is a document id (the `documents.document_id` primary key), not a chunk id.
6. Route order: `GET /documents/` is declared before `/{document_id}`, and `/upload` is POST only. No conflicts.
7. Several python processes are running, including `.venv\Scripts\python.exe` and `backend\.venv\Scripts\python.exe`. I did not map which one serves port 8000.

## Conclusions and recommendations
Root cause: an environment mismatch. The server is bound to `scatterbrain_huggingface.db`, while the requested document lives in `scatterbrain.db`. It is probably a leftover from switching to the HuggingFace embedding provider, which has its own dimension, so a separate DB was probably used.

Options (nothing implemented):
- Query an id that exists in the active DB, for example `644dd17d-8de8-4501-b7b5-cd6b019f3cda` (completed).
- To use the old data, restart the server with `SQLITE_DB_PATH=scatterbrain.db`, or clear the stray env var. Check the embedding provider and dimension first: the vectors in `scatterbrain.db` were probably made with a different model, so mixing them would break search.
- Hardening: resolve `sqlite_db_path` against the backend directory in `config.py`, as is already done for `.env`, so the cwd cannot change the DB. Log the resolved DB path at startup.
- Optionally return a clearer client-side error. The 404 detail is already explicit.
