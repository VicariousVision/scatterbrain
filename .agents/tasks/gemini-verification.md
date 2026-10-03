# Gemini provider verification (iteration 1)

Env: project venv `backend/.venv` (Python with pytest/fastapi). `google-genai==2.28.0` installed in it (also in root `.venv`).

SDK facts verified in the installed package: `errors.APIError(code, response_json, response)` has `code`, `status`, `message`, `details`; `HttpOptions.timeout` is milliseconds; `types.HttpRetryOptions` exists (not used; manual retry instead); `client.aio.models.generate_content` and `.get` used.

Commands run:
- `cd backend; .venv\Scripts\python.exe -m pytest -q` -> `98 passed, 1 skipped` (skip = opt-in RAGAS test). A multiprocess ResourceTracker traceback prints at interpreter exit; it is a teardown warning unrelated to the change.
- `cd backend; .venv\Scripts\python.exe -c "import main; ... create_chat_client(settings, object())"` with default env -> `OK ollama qwen3.5:0.8b Scatterbrain RAG API` (import and factory wiring fine, default provider ollama).

Not done: no real Gemini call, `.env` not read or edited (git shows `.env` modified from before this work; untouched by me). Existing chat tests unchanged; new tests only appended.
