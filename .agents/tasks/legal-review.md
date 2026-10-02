# Independent semantic review — legal/manual chunking

## Scope and evidence reviewed

I reviewed the current implementation against `.agents/tasks/legal-plan.md`, `.agents/tasks/authorised-dealers-chunking.md`, and `.agents/tasks/legal-verification.md`. I traced the production contracts through parser/cleaner, legal and generic chunking, SQLite migration/persistence, document lifecycle, retrieval, chat/citations, routers, Streamlit, the smoke harness, tests, and the legal-chunking guide.

I did not rerun the deterministic suite: the verification note records a successful 65-test backend run, compilation, a 298-page structural audit, and a temporary sqlite-vec retrieval sample. Those are useful baseline signals, but the tests do not exercise the defects below. No extra spot-check was needed because each defect follows directly from reachable control flow, and the required live-smoke artifact is absent.

## Sound parts of the implementation

The implementation preserves the project’s lifespan/module-level service wiring and does not introduce `Depends`. The settings relationships are validated, LangChain is exactly pinned, legal children—not parents—are inserted into `vec_chunks`, the legacy chunk migration is additive, embeddings are generated before the replacement transaction, and the old chat/search fields remain available. The parser preserves one-based PDF pages and visual prose/table ordering, the legal path enforces the 1,400-character source-text ceiling in the audited manual, and the Streamlit additions tolerate citation-free responses/history.

## Findings

### F1 — High: lowercase nested markers that are also Roman glyphs are assigned the wrong hierarchy

**Location:** `backend/services/legal_chunker.py`, `_marker_level()` and `_parse_hierarchy()`.

`_marker_level()` checks `_ROMAN_RE` before handling a single lowercase marker. Consequently `(c)`, `(d)`, `(l)`, `(m)`, `(v)`, and `(x)` enter the Roman branch even when they are peers of nested `(a)` and `(b)`. With realistic indentation, `(c)` can become a child of `(b)`; with shallow indentation it can become a Roman peer of `(i)`. The design evidence specifically identifies the real manual’s `A.3(B)(xxii)(a)`–`(c)` sequence, so this is not merely a synthetic edge case. It corrupts hierarchy, logical paths, parent attachment, embedding breadcrumbs, and citations.

Use expected tier, indentation, active ancestors, and marker-sequence context together rather than deciding from Roman spelling first. Add regressions for `(i) -> (a), (b), (c)` and other ambiguous glyphs at both shallow and realistic PDF indentation, asserting the complete canonical paths and attachment.

### F2 — High: exact nested/parent clause queries and cross-references cannot reliably resolve the stored child that contains them

**Locations:** `backend/services/legal_chunker.py`, `_primary_child_nodes()` / `_child_records_for_node()`; `backend/services/vector_store.py`, `get_children_by_clause_paths()`; `backend/services/retrieval_service.py`, `_rank()`.

The chunker normally emits Roman provisions as searchable children. If a Roman provision is structurally split at `(a)`/`(1)`, every resulting child still receives the Roman node’s `clause_path`; the contained nested path is not indexed as an alias. Conversely, a lettered target such as `I.3(B)` may have only descendant children such as `I.3(B)(i)`. `get_children_by_clause_paths()` then requires normalized equality with a child’s single `clause_path`, and the `+1.0` boost also requires equality.

As a result, exact questions about `B.2(B)(i)(a)` and parent-level references such as the design’s `I.3(B)` can fail both deterministic boosting and target expansion even though the containing source is stored. The current retrieval test uses a fake target whose child path is already exactly `I.3(B)`, so it masks this seam.

Persist canonical contained-clause aliases (or equivalent indexed hierarchy edges), assign a nested split its actual sub-item path, and resolve a requested governing path to a bounded, deterministic set of exact/descendant children in the same document. Add tests where `I.3(B)` has Roman descendants and where an overlong Roman provision splits into nested items. A reference string must remain non-evidence unless one of those actual target children is loaded.

### F3 — High: deleting a newly uploaded processing document can resurrect orphan vectors

**Location:** `backend/services/document_service.py`, `upload()`, `_process_document()`, and `delete_document()`.

`upload()` creates the metadata row and schedules `_process_document()` with `asyncio.create_task()` without registering the task or acquiring the lifecycle semaphore. An immediate delete can acquire the semaphore first, remove the metadata/vector state, and return. The still-pending ingestion task can then acquire the semaphore, insert parents/children/vectors for the deleted ID, fail to update the now-missing metadata row silently, and even execute same-filename supersession cleanup. The result is searchable data that is absent from document management, and it can replace a prior valid same-filename ingestion.

Track ingestion tasks and cancellation/tombstone state, and re-check the document lifecycle inside the semaphore before persistence. Delete should cancel and await a pending ingestion (or atomically mark it deleted so it cannot commit) before removing records. Add a deterministic event/barrier test in which delete wins the semaphore before the scheduled ingestion starts, then assert that documents, parents, children, and vectors are all absent.

### F4 — High: governing parent text is supplied as evidence under the child’s citation and page range

**Locations:** `backend/services/vector_store.py`, `get_parent_excerpt()`; `backend/services/retrieval_service.py`, `retrieve()`; `backend/services/chat_service.py`, `_format_document()` and `_citations()`.

For every primary child, retrieval takes the first 600 characters of the parent, regardless of the selected child’s position, and chat places that text inside the same numbered `<document>` block as the child. Citation construction nevertheless uses only the child metadata. For a later provision such as `(ii)`, the supplied excerpt can come from `(i)` or an earlier PDF page. The model may therefore make a supported statement from text it actually saw while the API reports the wrong clause/page. The excerpt can also end mid-sentence.

Either limit governing context to non-evidentiary heading/breadcrumb metadata, or model a parent excerpt as a separately provenance-bearing context with its own source spans/page range/citation. If it remains in the child block, select an ancestor excerpt around the child and make the citation metadata cover every quoted source segment. Add a cross-page parent test proving that every source string sent to the model has a truthful citation.

### F5 — High: `raw_text` and source spans are not immutable, faithful provenance

**Locations:** `backend/services/legal_chunker.py`, `_document_lines()`, `_child_records_for_node()`, `_front_matter_records()`, and table/code rendering.

The cleaner may rejoin `inter-\nnational`, reducing the number of clean lines, but `_document_lines()` pairs clean and raw text by the same list index. Every following raw line is then associated with the wrong clean line. In addition, every `_Line` reuses the block’s first whole-block `SourceSpan` rather than line/character offsets, and split children set `raw_text` to the normalized split piece. Front-matter and table records similarly label reconstructed/cleaned renderings as raw text. This contradicts the documented parser-to-store contract and makes audit/excerpt provenance unreliable even when page numbers happen to be correct.

Carry offset-aware source mapping through normalization, preserve original block/row slices as raw data, and keep reconstructed normalized text only in `source_text`. Split-child spans should identify the actual intersecting raw offsets. Add a dehyphenation-plus-following-lines regression and assertions that raw text and spans round-trip to the extracted block while source text remains normalized.

### F6 — High: multi-page tables lose required continuation semantics and legal location

**Locations:** `backend/services/legal_chunker.py`, `_stitch_tables()`, `_table_records()`, and `_apply_table_piece_provenance()`.

Table stitching only concatenates rows when adjacent tables have equal normalized headers. It has no implementation for forward-filling visually unambiguous continuation cells, even though the plan requires it. A blank leading continuation cell can therefore be packed into a different child without the governing key. Such a row also makes `_apply_table_piece_provenance()` fall back to every page in the table group. Further, table records discard the nearest table title/clause hierarchy and use only the running section plus the generic heading `Table`, weakening exact retrieval and citation.

Retain row/block geometry while stitching, distinguish merged/continued cells from genuinely blank values, forward-fill only demonstrably unambiguous keys, and preserve the nearest legal clause/table title in metadata and breadcrumbs. Keep row-level page provenance after grouping. Add a repeated-header, multi-page fixture with a continued blank key crossing a child boundary and assert meaning, path/title, and exact pages.

### F7 — High: deterministic content IDs collide when identical bytes are uploaded under different filenames

**Locations:** `backend/services/legal_chunker.py`, `_stable_id()`; `backend/services/vector_store.py`, `add_document()` and schema primary/unique keys.

Parent/child IDs intentionally omit upload UUID and filename, while `document_parents.parent_id`, `document_chunks.chunk_id`, and `vec_chunks.chunk_id` are globally unique. Replacement deletes only rows with the incoming filename. Therefore, after ingesting `manual.pdf`, uploading the same bytes as `manual-copy.pdf` generates identical IDs, deletes nothing, and fails on the existing primary/unique keys. The pre-existing upload API did not impose content uniqueness, so this is a backward-compatibility regression.

Separate stable content identity from the per-ingestion storage/vector key, or explicitly deduplicate identical sources with document aliases and safe reference-counted deletion. Add a test that ingests identical bytes under two filenames and verifies the documented behavior and independent deletion without corrupting the surviving document.

### F8 — High: the requested real passage-to-answer smoke verification was not completed

**Locations:** `.agents/tasks/legal-verification.md`; expected `.agents/tasks/authorised-dealers-smoke.md`; `backend/tools/authorised_dealers_smoke.py`.

The verification note explicitly says the live Ollama/API smoke was not run, and the required durable smoke report does not exist. The recorded retrieval sample uses an all-equal fake embedding provider and does not generate an LLM answer. This does not satisfy the user’s request or plan item 10 to ingest the selected real passage, query the actual system, and verify the grounded answer/citation.

Run the documented isolated local harness with the exact configured models and retain the report. Before running it, strengthen the assertion: `any(verify, verification, proof)` allows an answer containing only `proof` to pass, whereas the selected source and plan require Financial Surveillance Department verification **and** proof of bona fide nature/legitimacy for excess transfers. The report should capture both assertions along with the temporary/user-DB integrity evidence.

## Documentation follow-up

The dedicated guide is substantial and explains the rationale, migration, settings, and fallback. It must be corrected together with the findings above: it currently promises immutable raw text, canonical cross-reference resolution, and hierarchy disambiguation that the code does not provide. Its verification section also says pages `42–49` were sampled, while the verification note reports `[46, 47, 48, 49, 98, 99]` and the live harness is configured for `[47, 48, 49, 98, 99]`. Keep one accurate statement and distinguish the structural audit, fake-provider retrieval sample, and real live smoke.

## Gate basis

The successful deterministic suite and 298-page size/ID audit do not cover semantic path correctness, lifecycle races, evidence-to-citation provenance, duplicate-source storage, or the required live answer. These findings affect core hierarchy, data integrity, retrieval, and legal citation behavior and need correction plus focused regressions before the implementation can pass the review gate.
