# Tools and Tests

## Deterministic suite

From `backend/`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

No deterministic legal-ingestion test contacts Ollama. Temporary real SQLite/sqlite-vec databases use fake deterministic embedders.

| Test area | Coverage |
| --- | --- |
| `test_config.py`, `test_content_models.py` | defaults/relationships, structured JSON round trips, raw/source/embedding separation |
| `test_document_parser.py`, `test_text_cleaner.py` | page/source provenance, furniture, printed page/revision, front matter/main TOC/section indexes, mixed prose-table order, indentation |
| `test_legal_chunker.py`, `test_text_chunker.py` | hierarchy, cross-page clauses, exceptions, definitions, hard max, forced-only overlap, multi-page tables, BOP codes, generic LangChain fallback |
| `test_vector_store.py`, `test_document_db.py` | additive legacy migration, full metadata, parent-only storage, rollback, replacement/deletion |
| `test_retrieval_service.py` | exact identifier/amount ranking, deterministic diversification, continuation/list and cross-reference target expansion |
| `test_chat_service.py`, `test_api_citations.py` | context budget/security, typed results, legal labels, additive API compatibility |
| `test_frontend_citations.py` | printed/PDF ranges, revisions, fallback and duplicate safety |
| `test_document_service.py` | structured orchestration and failed replacement behavior |
| `test_ragas_adapters.py` | local RAGAS adapters and fictional corpus |
| `test_ragas_evaluation.py` | opt-in live local evaluation only |

Syntax/import validation from repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m compileall -q backend frontend
.\backend\.venv\Scripts\python.exe -c "import ast,pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('frontend').rglob('*.py')]"
```

## Real manual sample without Ollama

Implementation verification selects local PDF pages 42–49 and 98–99 with `parse_document_structured(..., pdf_pages=[...])`, then runs clean → legal chunk → temporary sqlite-vec → `RetrievalService` using a deterministic fake embedder. This verifies page/revision provenance, `B.2(B)(i)` cross-page grouping, 1,400-character enforcement, and exact `B.4(A)(i)` retrieval without changing the configured DB. The PDF and temporary DB are not committed.

The separate live-Ollama smoke step must use only `http://localhost:11434`, an explicit temporary `VectorStore(db_path=...)`, and pre/post fingerprints of the configured user DB. It should ask the documented `B.4(A)(i)` allowance/proof question and record citations; it is intentionally not part of the deterministic suite.

## RAGAS (opt-in)

```powershell
cd backend
pip install -r requirements-eval.txt
$env:RUN_RAGAS_EVAL = "1"; .\.venv\Scripts\python.exe -m pytest -m ragas -s
python -m evaluation.ragas_eval --output ragas_report.json
```

RAGAS reuses the structured production path and an explicit throwaway database. Its fictional TXT corpus still routes through generic `CHUNK_SIZE/CHUNK_OVERLAP`. Telemetry is disabled and all model calls target local Ollama.

## `visualize_embeddings.py`

```powershell
python visualize_embeddings.py --db scatterbrain.db --method pca --out-dir .
```

This development utility reads child vectors joined to `document_chunks` and plots them. Optional plotting dependencies are not runtime requirements. Parents are intentionally absent from `vec_chunks`.
