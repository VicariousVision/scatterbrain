# Legal/manual ingestion and retrieval

Scatterbrain uses a page-aware, hierarchy-aware, clause-first parent/child strategy for the **Currency and Exchanges Manual for Authorised Dealers** and structurally similar legal manuals. Ordinary TXT files and unrelated PDFs continue to use LangChain's `RecursiveCharacterTextSplitter` fallback.

## Why flat windows are unsafe

A fixed character window can separate an obligation from its amount limit, eligibility test, documentary requirement, exception, or `provided that` qualifier. It can also start midway through a sentence because of blind overlap. The manual compounds this problem with A–K sections, nested clause markers, definitions spanning subconditions, provisions continuing onto later pages, multi-page tables, BOP code lists, running page furniture, a version-control sheet, and an eight-page table of contents.

The searchable unit is therefore a **logical provision**, not a PDF page or arbitrary window. For example, `B.4(A)(i)` is retrieved as a child with its nested conditions and an inherited breadcrumb; its larger `B.4(A)` governing subsection is stored separately and is never vector-ranked.

## Pipeline and data contract

```text
bytes + filename
  -> parse_document_structured() -> ParsedDocument(pages -> ordered blocks)
  -> clean_parsed_document()     -> same geometry/raw text, populated clean text
  -> chunk_parsed_document()     -> parent/intermediate-parent/child ChunkRecords
  -> VectorStore.add_document()  -> parents + child metadata + child vectors
  -> RetrievalService.retrieve() -> boosted/diversified children + needed expansion
  -> ChatService.query()         -> numbered source blocks + typed citations
```

`parse_document()` and `clean_text()` remain compatibility wrappers. `chunk_text()` remains the generic splitter. Structured code keeps three text forms distinct:

- `raw_text`: extracted source before normalization;
- `source_text`: normalized quotation used in answers and citations;
- `embedding_text`: inherited breadcrumb/legal location plus source text, used only for embeddings.

Breadcrumbs improve retrieval without corrupting quoted source text.

## Page-aware extraction

`parse_document_structured(filename, content, pdf_pages=None)` returns `ParsedDocument` with one-based PDF pages. `pdf_pages=[47, 48, 49]` selects an explicit subset, which is useful for isolated local tests without renumbering the pages.

Each `ExtractedPage` records PDF page, detected printed page, page revision, running section marker, page classification, dimensions, and ordered `ExtractedBlock` objects. Each block records its bounding box, order, extraction method, source spans, and table structure where relevant.

For PDFs, the parser:

1. detects visual tables and excludes words inside their bounding boxes from prose;
2. groups remaining positioned words into lines/nearby prose blocks;
3. sorts prose and table blocks by `(top, x0)`, preserving prose → table → prose order on mixed pages;
4. analyzes top/bottom coordinate bands across pages to remove repeated title/page-number furniture;
5. retains the section marker from the header and printed page/revision from the footer as metadata;
6. classifies version-control pages as `front_matter` and both the main multi-page contents and section-level multi-page indexes as `navigation`, retaining version/navigation metadata but not embedding those pages.

Detection is conservative for unrelated PDFs. TXT becomes one generic page/block.

## Hierarchy grammar and boundaries

The stateful legal parser carries active hierarchy across page boundaries and recognizes:

- top-level sections `A.`–`K.` and numbered sections such as `B.2`;
- lettered subsections `(A)`;
- Roman provisions `(i)`, `(ii)`, …;
- nested lower-case `(a)`, `(aa)` and numeric `(1)` items.

Indentation and current context disambiguate markers such as nested `(i)`. A peer-or-higher marker closes the current node; a page boundary does not. Plain continuation language—including exceptions, amount limits, eligibility, reporting/documentary duties, and `provided that` text—stays attached to the deepest active provision.

### Child and parent sizing

- Searchable child target: **900 characters**.
- Soft intact range: **450–1,100 characters**. Short complete provisions remain intact.
- Strict searchable hard maximum: **1,400 characters**.
- Normal overlap: **zero** between independent clauses/rows.
- Forced continuity tail: **120 characters**, only when one continuous provision must be sentence/recursive-split.
- Governing parent: normally the lettered subsection/top-level rule, retained unembedded. Above **8,000 characters**, Roman provisions also receive intermediate parents.

An overlong provision first splits at legal sub-items. Tables split at complete rows. Only when no legal/row boundary can satisfy the maximum does the recursive sentence fallback run. Forced pieces get stable `continues_from`/`continues_to` links. Independent provisions never inherit those links or the continuity tail.

## Structured special content

- **Definitions:** one child per defined term, including all nested statutory tests/subconditions. Distinct stable IDs include the normalized term.
- **Tables:** compatible adjacent-page tables with the same real header are stitched before grouping. Every searchable row group repeats that header and preserves complete rows. A pathological overlong row remains complete in its parent and becomes linked sentence-bounded children that repeat the true header and row key.
- **BOP/code lists:** code and description stay together. Dense code rows become `code_list` parents/children; long descriptions repeat their actual code in forced pieces.
- **Schedules/forms:** actual supplied content can be typed as schedule/form content. A textual reference to an absent schedule, specimen, or form does not cause invented content.
- **Cross-references:** canonical targets such as `A.3(B)(xxii)` and `I.3(B)` are extracted to metadata. A reference string alone is not treated as proof.

## Stable metadata and SQLite migration

Children and parents include source hash/version, PDF and printed page ranges, revisions, section/clause paths, heading/breadcrumb, parent/child IDs and order, content type, continuation links, table/header/code/term fields, cross-references, parser/extraction versions, spans, and separate raw/source/embedding text.

`VectorStore.initialize()` is additive and idempotent:

- existing `document_chunks` and `vec_chunks` are not dropped or rebuilt;
- missing structured columns are added with `PRAGMA table_info` checks;
- legacy flat rows copy `content`/`chunk_id` into compatibility fields and remain searchable with `is_structured=0`;
- unembedded records live in `document_parents`; only child IDs exist in `vec_chunks`;
- provider/model/dimension mismatches stop startup without deleting data;
- `VectorStore(db_path=...)` supports isolated test/evaluation databases.

Legacy rows do not have page/hierarchy provenance and must be **re-ingested** to gain structured metadata. `DocumentDB` marks old completed rows with `needs_reingestion=true`. Uploading the same filename computes embeddings first, then replaces parents, children, and vectors in one `BEGIN IMMEDIATE` transaction. Failure rolls the replacement back. Superseded document metadata is removed only after success. Explicit deletion removes parent, child, and vector rows in one transaction.

Do not delete a production database merely to apply this schema migration. A new database/re-ingestion is required only when intentionally changing the embedding provider/model/dimension.

## Hybrid retrieval and context expansion

Retrieval ranks **children only**. It fetches a wider candidate pool (default 10), then deterministically adds:

- `+1.00` for an exact canonical clause identifier;
- `+0.35` for an exact amount or BOP code;
- `+0.20` for a defined term/acronym.

Ties retain vector rank and then child ID. Selection defaults to five children, initially capped at two per parent and three per section; caps relax only if needed to fill the requested count.

For selected children, Scatterbrain may add a short governing parent excerpt. It loads adjacent children only for explicit continuation links or list-oriented questions. Canonical cross-reference targets are resolved to actual stored children and included as `cross_reference` evidence; the prompt explicitly forbids relying on unresolved textual references. Full parents are never appended blindly.

`ChatService` formats only contexts that fit the model budget and returns citations for exactly those contexts. Structured labels use clause/section and printed page with PDF fallback, for example `B.4(A)(i), p. 98`; legacy chunks fall back to filename/chunk. `ChatResponse.response`, `history`, and integer `retrieved_chunks` remain unchanged, with additive `citations`. Streamlit tolerates old backends/history and shows a compact Sources line when citations exist.

## Configuration

| Environment variable | Default | Purpose |
| --- | ---: | --- |
| `CHUNK_SIZE` | `1000` | Generic/final recursive fallback size only |
| `CHUNK_OVERLAP` | `200` | Generic fallback overlap only |
| `LEGAL_CHUNK_TARGET_CHARS` | `900` | Legal child packing target |
| `LEGAL_CHUNK_MIN_CHARS` | `450` | Soft intact minimum |
| `LEGAL_CHUNK_HARD_MAX_CHARS` | `1400` | Strict searchable legal maximum |
| `LEGAL_FORCED_SPLIT_OVERLAP_CHARS` | `120` | Forced continuous-provision tail only |
| `LEGAL_PARENT_MAX_CHARS` | `8000` | Intermediate-parent threshold |
| `RETRIEVAL_CANDIDATE_POOL` | `10` | Vector candidates before hybrid ranking |
| `RETRIEVAL_TOP_K` | `5` | Final diversified primary children |
| `RETRIEVAL_MAX_CHILDREN_PER_PARENT` | `2` | First-pass parent cap |
| `RETRIEVAL_MAX_CHILDREN_PER_SECTION` | `3` | First-pass section cap |

Settings validate positive/range relationships at startup. In particular, minimum ≤ target ≤ hard maximum, forced overlap < target/hard maximum, and candidate pool ≥ final top-k.

## Operation and verification

Install and run normally:

```powershell
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Re-upload the manual through the Upload page or `POST /documents/upload`; using the same filename performs rollback-safe replacement. Poll `GET /documents/{id}` until `completed`, then inspect `GET /search/?q=B.4(A)(i)` or ask in Chat.

Deterministic tests (no Ollama):

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q
cd ..
.\backend\.venv\Scripts\python.exe -m compileall -q backend frontend
```

The implementation verification also parses the real local pages 42–49 and 98–99 and runs them through a temporary SQLite/sqlite-vec database with a deterministic fake embedder. It does not copy the 298-page PDF or create a persistent vector database.

A separate live-Ollama step can run the isolated API smoke harness (it fingerprints the user DB, uses an explicit temporary DB, and removes it):

```powershell
cd backend
$env:EMBEDDING_PROVIDER='ollama'
$env:OLLAMA_BASE_URL='http://localhost:11434'
$env:OLLAMA_MODEL='qwen3.5:0.8b'
$env:OLLAMA_EMBEDDING_MODEL='nomic-embed-text'
$env:EMBEDDING_DIMENSION='768'
$env:OLLAMA_NUM_GPU='0'
.\.venv\Scripts\python.exe -m tools.authorised_dealers_smoke --pdf "..\Currency and Exchanges Manual for Authorised Dealers.pdf" --report "..\.agents\tasks\authorised-dealers-smoke.md"
```

This command intentionally is not part of the deterministic suite and was not run during implementation verification.

## Limitations

- Extraction uses the PDF text layer; scanned inserts need OCR before this parser can understand them.
- Header/footer and front-matter rules are conservative and tuned with fallback behavior; materially different manuals need representative geometry tests.
- Table stitching requires adjacent pages and compatible real headers. Ambiguous blank continuation cells are not guessed.
- Hierarchy detection is syntactic, not legal interpretation. Retrieved text should not be treated as professional legal advice.
- Existing flat ingestions stay searchable but cannot gain missing clause/page metadata without re-ingestion.
