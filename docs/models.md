# Models (`backend/models/`)

Pydantic API and internal transfer models only; no ORM/database behavior lives here.

## `models/content.py`

| Model | Purpose |
| --- | --- |
| `SourceSpan` | One-based PDF page, printed page, revision, bounding box, block order, optional character range |
| `ExtractedBlock` | Ordered coordinate-aware prose/table block with raw/clean text and table rows/header |
| `ExtractedPage` | Page provenance, body/front-matter/navigation classification, ordered blocks |
| `ParsedDocument` | Source hash/title/version, extraction/parser/schema versions, pages and navigation map |
| `ChunkRecord` | Parent/intermediate/navigation or searchable child with complete structured metadata |
| `RetrievedContext` | Ranked source child, relationship, short governing context, metadata |
| `Citation` | Additive API source/legal location and display label |
| `ChatResult` | Typed answer, exactly-used contexts, citations; still unpacks as historical `(answer, source_texts)` |

`ChunkRecord` keeps `raw_text`, `source_text`, and `embedding_text` separate. `record_type="child"` requires `child_id`; non-child records require `parent_id`. Fields include source SHA/version, page ranges/revisions, section/clause paths, breadcrumb, relationship IDs, table/code/term data, cross-references, extraction method, parser version, and source spans.

## `models/chat.py`

### `ChatRequest`

`query: str`; `history: list[dict] = []` (factory-backed). History remains display state and is not treated as retrieved evidence.

### `ChatResponse`

| Field | Type | Compatibility |
| --- | --- | --- |
| `response` | `str` | unchanged |
| `history` | `list[dict]` | unchanged |
| `retrieved_chunks` | `int` | unchanged count |
| `citations` | `list[Citation]` | additive, defaults empty |

## `models/document.py`

`DocumentRecord` retains ID, filename, upload time, status, and error, and adds chunk count, source hash, detected title/version, extraction/parser/schema versions, and `needs_reingestion` for legacy flat rows.

`UploadResponse.document_id` is unchanged. `DocumentListItem` exposes additive status/provenance fields.

`SearchResultItem` retains required legacy fields (`chunk_id`, `document_id`, `filename`, `chunk_index`, `text`, `similarity`) and adds score, child/parent IDs, source/version/page/revision/hierarchy fields, content relationships, table/code/term/cross-reference metadata, and governing/relation data. Old JSON still validates because every addition is optional/defaulted.
