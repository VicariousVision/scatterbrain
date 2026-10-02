# Services (`backend/services/`)

CPU-heavy parsing/chunking runs through `asyncio.to_thread`; embedding, sqlite-vec, retrieval, and Ollama calls are async. See [legal-document-chunking.md](legal-document-chunking.md) for full design rationale.

| Service | Responsibility |
| --- | --- |
| `DocumentService` | Hash/upload status, structured pipeline, serialized replacement/deletion |
| `DocumentDB` | Additively migrated document status/provenance metadata |
| `document_parser` | Page/block/coordinate-aware PDF and UTF-8 TXT extraction |
| `text_cleaner` | Conservative string and block cleaning without raw/layout loss |
| `legal_chunker` | Legal detection, hierarchy, parents/children, tables/codes/cross-refs |
| `text_chunker` | LangChain recursive fallback and generic Markdown table grouping |
| `VectorStore` | Parent/child persistence, child vectors, legacy migration, atomic cleanup |
| `RetrievalService` | Candidate pooling, lexical boosts, diversification, context expansion |
| `ChatService` | Context budgeting/security, generation, exactly-used citations |
| `EmbeddingProvider` | Ollama or Hugging Face document/query vectors |
| `OllamaClient` | Local async embeddings/generate/chat/health API |

## Structured ingestion

`DocumentService.upload` hashes the exact bytes and creates a persistent `processing` record. Its background operation acquires one semaphore shared with deletion, then calls:

1. `parse_document_structured(filename, bytes)` in a thread;
2. `clean_parsed_document(parsed)` in a thread;
3. `chunk_parsed_document(cleaned, document_id=...)` in a thread;
4. `VectorStore.add_document(...)` to embed children and atomically replace same-filename storage;
5. `DocumentDB.update_status(..., completed)` and then removes superseded same-filename metadata.

Parsing, embedding, or replacement failure marks only the new upload failed and does not proactively delete the prior ingestion. Explicit deletion removes vector records before status metadata under the same semaphore.

## Parser and cleaner

`parse_document_structured(filename, content, pdf_pages=None) -> ParsedDocument` supports one-based page selection and retains pages, printed labels, revisions, section markers, blocks, coordinates, tables, and source spans. Positioned prose is split around table boxes and sorted with tables by geometry. Repeated top/bottom furniture is removed after cross-page analysis. Version-control, main TOC, and section-level index pages remain unembedded metadata/navigation. `parse_document` flattens this contract for compatibility.

`clean_parsed_document` deep-copies and fills block `clean_text`; raw text, geometry, pages, rows, and spans remain unchanged. `clean_text` performs NFKC, line-ending, hyphenation, control/zero-width, and horizontal whitespace normalization while retaining first-line and nested indentation.

## Legal and generic chunkers

`legal_chunker.chunk_parsed_document` recognizes A–K/numbered sections, `(A)`, Roman, lower-case/double-lower-case, and numeric markers using state and indentation. It carries nodes across pages, keeps nested modifiers with the provision, emits stable unembedded parents/intermediate parents and children, extracts cross-references, stitches compatible multi-page tables, groups complete rows, and models BOP codes with descriptions.

Legal settings are 900 target, 450 soft minimum, 1,400 hard maximum, zero normal overlap, and a 120-character tail only for forced continuous-provision splits. `text_chunker.chunk_text` uses the pinned LangChain `RecursiveCharacterTextSplitter` for generic documents/final fallback; generic `CHUNK_SIZE/CHUNK_OVERLAP` never become legal peer-clause overlap.

## VectorStore

Constructor: `VectorStore(embedding_provider, db_path=None)`. An explicit path isolates tests/evaluation from user data.

Initialization loads sqlite-vec, additively migrates `document_chunks`, creates/migrates unembedded `document_parents`, preserves existing `vec_chunks`, fills compatibility fields on flat rows, and validates provider/model/dimension. It never drops/rebuilds a user database. Legacy completed records require re-ingestion for structured metadata.

Important methods:

| Method | Behavior |
| --- | --- |
| `add_document` | Accepts legacy strings or `ChunkRecord`s; computes child embeddings first, then transactionally replaces same-filename parents/children/vectors |
| `search_candidates` | Cosine-ranks only rows joined to `vec_chunks`, returning full additive metadata |
| `search` | Backward-compatible semantic child facade |
| `get_parent_excerpt` | Loads a short unranked governing excerpt |
| `get_adjacent_children` | Loads explicit/immediate same-parent relationships |
| `get_children_by_clause_paths` | Resolves actual canonical clause targets |
| `delete_document` / `delete_by_filename` | One transaction for child rows, parent rows, and vectors |

`embedding_metadata` protects fixed sqlite-vec dimensions. A mismatch raises without changing the database.

## RetrievalService

`retrieve(query, top_k=None, document_id=None)` gets the configured wider child pool, merges exact clause-path hits, adds fixed identifier/amount-or-code/term-or-acronym boosts, and deterministically diversifies by parent and section. It relaxes caps only to fill the final count. It conditionally loads short governing context, continuation/list adjacency, and actual cross-reference targets. Unresolved reference text is not exposed as target evidence.

`search(...)` serializes primary retrieved contexts for the raw search router.

## ChatService

`ChatService(vector_store, ollama_client, think=None, num_ctx=8192, retrieval_service=None)` uses hybrid retrieval when injected; omission adapts legacy vector search for compatibility/tests. It neutralizes document delimiter-like tags, escapes attributes, labels structured blocks with clause/page/revision/breadcrumb, includes only a short governing excerpt, and keeps a strict rank prefix within the model character budget.

`query` returns `ChatResult(answer, contexts, citations)`. Its iterator preserves old two-value unpacking. Citations are deduplicated by legal location and correspond only to contexts actually presented to the model. `OllamaClientError` propagates to the router as HTTP 503.

## Embeddings and Ollama

`create_embedding_provider` selects the local Ollama client or `HuggingFaceEmbeddingProvider`. Query-specific provider instructions are applied only to query vectors; document/embedding text remains unprefixed by that provider instruction.

`OllamaClient` sends `num_gpu` and `num_ctx`; provides `generate_embedding`, `generate`, `chat`, and `health_check`; serializes outbound work with `OLLAMA_MAX_PARALLEL` (default 1); and retries local connection failures. All endpoints remain local to configured Ollama.
