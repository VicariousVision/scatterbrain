# Implementation Plan: Gemini as optional chat LLM

Work directly in the workspace (no worktree). Tests: `cd backend; pytest`.
Source of truth for sketches: `.agents/reports/gemini-chat-investigation.md`.

## Constraints and corrections
- The agent `wf-reviewer` does NOT exist. The review step uses `semantic_reviewer`.
- NEVER read, print or edit real secrets in the root `.env`. Only edit `.env.example`.
- Default `LLM_PROVIDER=ollama`. Embeddings are unchanged.
- Verified `google-genai` latest = **2.28.0** (`pip index versions google-genai`). It is NOT installed locally yet, so item 1 installs it; verify SDK facts against the installed package (item 3).
- Report corrections: there is NO reranker (ranking is `RetrievalService._rank`/`_diversify`, already feeding the prompt). Embedding provider defaults to `huggingface` in config.py and .env.example (local .env uses ollama; do not read it). When `EMBEDDING_PROVIDER=ollama`, `create_embedding_provider` returns the OllamaClient, so OllamaClient must stay in main.py.
- Decision: not decomposed into FEATs. ~8 tightly coupled files; the existing implement/review loop runs this plan.
- Model: `gemini-3.5-flash-lite` (user request, free tier). Free tier may use prompts for Google product improvement; mention in .env.example comment.

## Items (order = dependency)

- [ ] 1. Add dependency. `backend/requirements.txt`: append `google-genai==2.28.0`. Then `pip install google-genai==2.28.0`.
      Files: backend/requirements.txt
      Verify: `python -c "from google import genai; from google.genai import types, errors; print(genai.__version__)"` prints 2.28.0.

- [ ] 2. Verify SDK facts in the installed package before coding the client (record results in the module docstring/comments):
      `errors.APIError` attributes (`code`, `message`, `status`, `details`, response headers / retry delay location), `types.HttpOptions(timeout=...)` units (expected ms), `client.aio.models.generate_content` exists, `types.HttpRetryOptions` availability, `client.aio.models.get`.
      Verify: `python -c "import inspect; from google.genai import errors, types; print(inspect.signature(errors.APIError.__init__)); print(types.HttpOptions.model_fields['timeout'])"`.

- [ ] 3. Shared error base. Create `backend/services/llm_errors.py` with `class LLMClientError(Exception)`. Make `OllamaClientError` (services/ollama_client.py line ~107) subclass it (import it there).
      Files: backend/services/llm_errors.py, backend/services/ollama_client.py
      Verify: `pytest tests/test_ollama_client.py tests/test_chat_service.py` still pass.

- [ ] 4. Config. In `backend/config.py` add to `Settings`: `llm_provider: str = "ollama"`, `gemini_api_key: SecretStr = SecretStr("")`, `gemini_model: str = "gemini-3.5-flash-lite"`, `gemini_temperature: float = Field(0.1, ge=0, le=2)`, `gemini_max_output_tokens: int = Field(1024, gt=0)`, `gemini_timeout_seconds: float = Field(60.0, gt=0)`, `allow_cloud_llm: bool = False`, `gemini_context_tokens: int = Field(16000, gt=0)` (moderate for free-tier TPM). In `validate_related_limits` add: lowercase/validate provider in {ollama, gemini}; if gemini, raise ValueError unless `gemini_api_key.get_secret_value().strip()` AND `allow_cloud_llm`. Never include key in error text.
      Files: backend/config.py
      Verify: new tests in item 9; `pytest tests/test_config.py`.

- [ ] 5. GeminiClient. Create `backend/services/gemini_client.py` with `GeminiClientError(LLMClientError)` and `GeminiClient(api_key: str, model, timeout, temperature, max_tokens)`. Follow report sketch: `provider_name="gemini"`; `genai.Client(api_key, http_options=types.HttpOptions(timeout=int(timeout*1000)))`; async `chat(messages, *, json_schema=None, max_tokens=None, think=None, temperature=None) -> str` (same signature as OllamaClient.chat; `think` accepted and ignored); system messages joined into `system_instruction`; `assistant`->`model`; consecutive-role safe. Free-tier handling: on `errors.APIError` with code 429/503, retry up to 3 attempts with exponential backoff (1s,2s,4s; cap ~30s), honoring the API-provided retry delay when present (from item 2 findings; use an injectable `_sleep` so tests do not wait). On final 429 raise `GeminiClientError("Gemini quota or rate limit exhausted (free tier limits); wait and retry or check AI Studio quota")`. Other APIError -> `GeminiClientError(f"Gemini API error {code}")` (no key/prompt in messages or logs). Empty `resp.text` -> GeminiClientError with finish reason. Also catch generic transport exceptions (httpx/timeouts) -> GeminiClientError. `async health_check() -> bool` via `aio.models.get(model=...)`, swallow errors, return False.
      Files: backend/services/gemini_client.py
      Verify: item 9 tests.

- [ ] 6. Gemini prompt builder. Create `backend/services/gemini_prompt.py` exposing `GEMINI_SYSTEM_PROMPT` (report's 8-rule layout; exact refusal sentence `I can't find this in the provided documents.`; `[n]` citation rule using block ids; untrusted-data rule) and `build_gemini_messages(blocks_or_results, query)` returning `[system, user]` with documents first, question last. Document format: `<document id="n" relation="..." source="..." clause="..." printed_page="..." chunk="...">` + optional governing context + `Text: ...`; reuse `_escape_attr`, `_neutralize`, `format_citation_label` from chat_service (move nothing; import lazily or copy helpers into the new module to avoid circular imports; prefer importing from chat_service only if no cycle). Export a `select_within_budget`-compatible formatter so ChatService can budget with it.
      Files: backend/services/gemini_prompt.py
      Verify: item 9 prompt test.

- [ ] 7. ChatService generalization. In `backend/services/chat_service.py`: ctor becomes `__init__(self, vector_store, llm_client=None, think=None, num_ctx=8192, retrieval_service=None, *, ollama_client=None, prompt_style="ollama")`; `llm_client = llm_client or ollama_client` (raise TypeError if neither); keep positional order so existing `ChatService(vector_store=..., ollama_client=client, ...)` works; store as `self._llm_client` (keep `_ollama_client` alias property if any test touches it). When `prompt_style == "gemini"`, use gemini_prompt formatter/system prompt for budgeting (fixed-char size) and messages; otherwise the existing Ollama prompt is untouched. Replace `except OllamaClientError` with `except LLMClientError` (log "LLM generation failed", re-raise). Update docstring (NumPy style).
      Files: backend/services/chat_service.py
      Verify: `pytest tests/test_chat_service.py` passes UNCHANGED.

- [ ] 8. Router, main, health, env example.
      - `backend/routers/chat.py`: catch `LLMClientError`; 503 detail `"The LLM service is unavailable. Please try again later."`; update docstring. Check `tests/test_api_citations.py` for any assertion on the old "Ollama" detail and update only if needed.
      - `backend/main.py`: add `create_chat_client(settings, ollama_client)` factory returning `(client, num_ctx, think, prompt_style, provider_name)`; gemini branch builds GeminiClient with `settings.gemini_api_key.get_secret_value()`; ollama branch returns existing values. Keep OllamaClient creation, health check and embedding provider wiring. Log `Chat answers provided by: %s (model %s)` (no key). Pass to `ChatService(llm_client=..., ...)`. Store provider info on `app.state.llm_provider` / `llm_model`.
      - `backend/routers/health.py`: add `llm_provider` and `llm_model` to the response (read from `request.app.state` with defaults "ollama"); keep `"status": "ok"`.
      - `.env.example`: add commented block `# LLM_PROVIDER=ollama`, `# GEMINI_API_KEY=`, `# GEMINI_MODEL=gemini-3.5-flash-lite`, `# ALLOW_CLOUD_LLM=false` with a privacy note (retrieved text is sent to Google; free tier data may be used by Google).
      Files: backend/routers/chat.py, backend/main.py, backend/routers/health.py, .env.example
      Verify: `pytest` (full) passes; `python -c "import main"` from backend imports cleanly.

- [ ] 9. Mock-only tests (no network, no real key):
      - `backend/tests/test_gemini_client.py`: patch `client._client.aio.models.generate_content` with AsyncMock; assert system -> `system_instruction`, assistant -> role `model`, `think` ignored, json_schema sets mime type, APIError -> GeminiClientError, empty text -> error, 429 retried then succeeds (with no-op `_sleep`), 429 exhausted -> quota message, health_check False on error. Build `errors.APIError` instances per item 2 findings.
      - `backend/tests/test_config.py`: add cases: gemini without key fails; gemini with key but `allow_cloud_llm=False` fails; gemini with both passes; default provider is ollama. Use `Settings(_env_file=None, ...)`.
      - `backend/tests/test_chat_service.py`: ADD (do not alter existing) a test that `GeminiClientError` propagates and that `llm_client=` and `ollama_client=` both work; a router test (in test_api_citations.py style or new file) that a GeminiClientError maps to 503 generic detail.
      - `backend/tests/test_gemini_prompt.py`: refusal sentence present, documents precede question, attributes id/relation/source/clause/page present, delimiter text neutralized, ollama prompt unchanged.
      Verify: `cd backend; pytest tests/test_gemini_client.py tests/test_config.py tests/test_chat_service.py tests/test_gemini_prompt.py -v`, then `pytest` full suite green.

## Final check
`cd backend; pytest` green; `git status` shows no `.env` change; default provider still ollama.
