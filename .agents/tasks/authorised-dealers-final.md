# Authorised Dealers manual — final build record

## Build and documentation

The existing page-aware hierarchy/clause-first parent-child implementation was retained. Its working patch passed `git diff --check`; no broad test suite or repeat 298-page audit was run, as requested.

Current modified project files (including prior implementation work):

- `.env` — local application selection changed to Ollama embeddings and `scatterbrain.db`.
- `.env.example`
- `README.md`
- `backend/services/document_parser.py`
- `backend/services/legal_chunker.py`
- `backend/services/retrieval_service.py`
- `backend/tests/test_document_parser.py`
- `backend/tests/test_legal_chunker.py`
- `backend/tests/test_retrieval_service.py`
- `backend/tests/test_vector_store.py`
- `backend/tools/authorised_dealers_smoke.py`
- `docs/legal-document-chunking.md`
- `docs/services.md`
- `docs/tools-and-tests.md`

The documentation now explicitly states that normal uploads persist metadata, parents, searchable children, and vectors in `SQLITE_DB_PATH`; same-filename uploads transactionally replace the prior ingestion, and provider/model/dimension changes require a separate database.

## Persistent database evidence

- Configured application database: `C:\Users\ozzey\Documents\git-projects\scatterbrain\backend\scatterbrain.db`
- Pre-mutation backup: `C:\Users\ozzey\Documents\git-projects\scatterbrain\.agents\backups\scatterbrain-20261002-201152.db`
- Embedding configuration: `ollama` / `nomic-embed-text` / 768 dimensions
- Generation model: `qwen3.5:0.8b`
- Document ID: `dfa8e4cc-f66b-423c-9f1b-f83dfde7b41b`
- Status: `completed`
- Stored parents: 512
- Stored children: 1,270
- Stored vectors joined to this document: 1,270
- Reported document chunk count: 1,270
- Source SHA-256: `ff3adcf9bb62e8b0f0d7686e464c6c70cceb3afadb6b74ca75076383af6625dc`
- Detected version: `1.131 (2026-04-29)`

The unrelated Hugging Face database (`backend/scatterbrain_huggingface.db`) was not mutated or deleted. Existing records in the Ollama database were preserved except any prior records with this exact manual filename, which the normal same-filename replacement path supersedes.

## One live persisted query

Question:

> Under B.4(A)(i), what is the R2 million allowance and what verification or proof is required for current transfers above it?

Result: `B.4(A)(i)` ranked first. The local model answered that the R2 million amount is the annual Single Discretionary Allowance limit for residents aged 18 or older, and that transfers above it require Financial Surveillance Department verification plus proof of bona fide nature and legitimacy.

Primary citation: `B.4(A)(i), p. 98 · rev. 6/2026` (child `c_c9b379f218980087ef8a2ef9`).

## Blockers

None. A known third-party `multiprocess.resource_tracker` destructor warning appeared after successful ingestion, as in prior verification; the ingestion process exited 0 and persisted counts/status were re-read afterward.
