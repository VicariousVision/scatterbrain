# Chunking findings — Currency and Exchanges Manual for Authorised Dealers

## Summary answer

**Use a page-aware, hierarchy-aware, clause-first parent/child strategy; do not ingest this manual with only generic recursive character splitting.** The manual is a 298-page regulatory instrument whose answers often depend on a complete numbered obligation plus its nested conditions, exceptions, amount limits, and cross-references. Make a logical provision (for example, `B.4(A)(i)` or `B.2(B)(i)(a)`) the primary boundary, prepend its inherited section breadcrumb to every searchable child, and retain a larger parent section for answer context.

Recommended parameters (the current application measures chunk size in **characters**, not tokens):

- **Searchable child target:** 900 characters (roughly 250–350 English tokens); accept an intact clause from 450–1,100 characters.
- **Child hard maximum:** 1,400 characters for prose or a table/code-list group. Split only on the next legal sub-item boundary (`(a)`, `(aa)`, `(1)`) or a table-row boundary. If no such boundary exists, split at a sentence boundary with a 120-character continuity tail.
- **Parents:** Store the complete lettered subsection or top-level rule as an unembedded parent (normally 2,000–8,000 characters; no strict retrieval-size limit). For an unusually large subsection, use an intermediate parent for the Roman-numeral provision.
- **Overlap:** zero for intact legal clauses and table/code-list groups. Use 120 characters **only** for a forced split of one continuous provision; do not use a blind overlap across separate clauses. Repeated breadcrumb, clause identifier, and table header are the meaningful replacement for overlap.

This preserves the rule and avoids an answer retrieving an exception, limit, or list item without the governing condition. It also keeps five retrieved children comfortably below the current 8,192-token chat context budget after source wrappers and answer reserve.

## Evidence

### PDF structure and extraction characteristics

- The PDF is a digitally generated Word document (metadata: title *Currency and Exchanges Manual for Authorised Dealers*, creator Microsoft Word for Microsoft 365) with **298 pages** and a selectable text layer. `pdfplumber` extracted 605,725 characters directly (mean 2,033/page; median 2,074; maximum 4,407), so OCR is not the first requirement.
- PDF pages 2–4 are a long version-control sheet, and pages 5–12 are an eight-page table of contents. The contents lay out a deep hierarchy: top-level `A`–`K`, section identifiers such as `B.2`, lettered children such as `(B)`, and many Roman/nested clauses. These front-matter pages should become document/version/navigation metadata, not ordinary semantic evidence chunks.
- Definitions span PDF pages 15–20 under `A.1`. They mix one-term definitions with nested statutory tests: for example, “Affected person” on p. 15 includes conditions `(i)` and `(ii)`. A definition must therefore keep its term and all qualification sub-items together.
- Duties on p. 24 begin `A.3(A)(i)`–`(vii)` and p. 28 shows `A.3(B)(xxii)` with nested `(a)`–`(c)`. This is direct evidence that a page or a 500-character window is not a safe logical unit.
- Long compliance rules cross pages. `B.2(B)(i)` begins at p. 47, continues across pp. 48–49, and contains amounts, eligibility, documentary requirements, reporting, exceptions, and cross-reference `I.3(B)`. `B.4(A)` starts at p. 98 and continues on p. 99 before lettered travel subrules start. Do not introduce a new semantic boundary merely because a PDF page ends.
- The manual includes dense numeric/code lists which are not all recognized as drawn tables: `J` pages 289–292 list BOP categories such as `511 01`–`511 08` and `701 01`–`705 02`. Treat these as structured code/description records, preserving each code with its description and hierarchy.
- `pdfplumber.find_tables()` detected 23 visual tables on pages 2–4, 21–23, 77, 140–141, 179, 198–201, 235–236, and 259–261. Examples: p. 21 is a one-column list of Authorised Dealers; pp. 198–201 are multi-page Authorised Bank/CSDP/settlement-agent lists; pp. 259–261 are a three-page `Concept | Description` table whose header repeats after the first page. Those page-spanning tables must be stitched before row-group chunking.
- The document has no `Annexure` or `Appendix` text hits. It does reference schedules (e.g., PDF pp. 46, 52, 56, 109, 134, 136, 170, 219) and external form/specimen material (notably pp. 25, 84–85, 295–296). Any supplied form/schedule should be kept as its own typed parent with a title/instructions child and separate field or row groups; do not merge it with preceding narrative.
- Every sampled content page has page furniture. On PDF p. 24 the running header is at vertical position 44–54 on an 842-point page, while the printed page number/revision is at 789–800. The header repeats the manual title and current section (`A.3` here; `B.2` repeats on 44 pages), and the footer carries page/revision values that differ by page. Remove the decorative manual-title/page-number portions by coordinates, but retain the section code as fallback navigation metadata and retain the revision date as `page_revision` metadata.

### Current repository behavior

- `backend/services/document_parser.py::_parse_pdf` extracts every page then joins pages with blank lines. `_extract_page` renders `find_tables()` output as Markdown and removes table-bbox words from prose, which is a useful starting point. However, it builds all non-table prose as one block and anchors it above the first table; on mixed-layout pages this can move prose that follows a table ahead of it. It also has no page number, coordinates, header/footer removal, or table-continuation metadata.
- `backend/services/text_cleaner.py::clean_text` preserves newlines and indentation, so it does not prevent a hierarchy parser from using the extracted layout. It currently lacks a page/section-aware normalization stage.
- `backend/services/text_chunker.py::chunk_text` recognizes only Markdown table blocks. All other material uses `\n\n`, `\n`, `. `, space, then character fallback in `_recursive_split`; it has no heading or legal-clause recognition. The configured defaults in `backend/config.py::Settings` are `chunk_size = 500` and `chunk_overlap = 100`.
- A read-only simulation through the installed parser → cleaner → current chunker produced **611,214 cleaned characters and 1,708 chunks** (median 462; mean 438; max 681; 64 table chunks). The number of chunks is not itself a failure, but 500-character windows will routinely separate a rule from a nested exception or cross-reference and the 100-character raw-tail overlap can start midway through a legal sentence.
- `backend/services/vector_store.py::VectorStore.initialize/add_document/search` persists only `document_id`, `filename`, `chunk_index`, and `content`; `search` returns those same minimal fields. `backend/services/chat_service.py::ChatService.query` retrieves a flat top-k (default five), and `_select_within_budget` keeps only a rank-ordered prefix. Neither component can currently expand a winning child to its governing clause/section or cite a legal location.

## Conclusions and recommendations

### Parsing and preprocessing

1. Produce a sequence of page objects, not one undifferentiated string: `{pdf_page, printed_page, page_revision, blocks[]}`. Crop y=0–about 60 and y=about 780–842 after confirming coordinates across representative page styles; remove the repeated title/page-number furniture while preserving a detected section marker. Preserve source-page boundaries internally even when a logical provision later spans them.
2. Convert the table of contents into a navigation map (`section_id`, title, printed start page) and version-control pages into document-level provenance. Exclude both from normal semantic embedding, unless a product requirement explicitly needs amendment-history queries.
3. Detect headings with the manual’s actual grammar: top-level `^[A-K]\.`; numbered `^[A-K]\.\d+`; lettered `^\([A-Z]\)`; Roman `^\([ivxlcdm]+\)`; nested `(a)`, `(aa)`, and numeric `(1)`. Keep the current line indentation because it helps disambiguate levels. Validate the hierarchy against the parsed table of contents, but do not rely only on the contents because amendments may move text.
4. Carry a provision across PDF pages until the next peer-or-higher heading. Prepend an inherited breadcrumb such as `B.4 > (A) Single discretionary allowance > (i)` to every child, even if the original continuation page only says `B.4` in its header.

### Semantic boundaries and special content

- **Definitions:** emit one child per defined term plus all of that term’s subconditions; include `section=A.1`, `term`, and normalized acronym aliases. Combine only adjacent short definitions when the target would otherwise be very small, never split term from definition.
- **Rules, conditions, and exceptions:** include an entire Roman clause and its nested items when it fits. For an over-limit clause, give every forced child the clause heading and an explicit `continues_from`/`continues_to` relation. Keep exception/“provided that” sentences with the condition they modify, even if that produces a child near the hard maximum.
- **Tables:** retain Markdown/structured rows, table title, section breadcrumb, and page span. Repeat the true column header in every child; use groups of complete rows under the 1,400-character maximum. Normalize multi-line cells and forward-fill blank continuation cells only when visually unambiguous. Stitch known continuation tables before grouping; do not assume every pipe-like list is a table.
- **Codes, schedules, and forms:** model code lists as `(code, label/description)` row children under a typed `code_list` parent. Model a schedule/form as `form_instructions` plus `form_field_group`/`form_row_group` children, retaining field labels and form identifier. Link an external form/specimen reference instead of inventing absent content.

### Metadata schema

Store this with every child (and a corresponding parent record), rather than only a positional `chunk_index`:

```text
document_id, document_title, source_filename, source_sha256,
document_version_or_issue_date, pdf_page_start/end, printed_page_start/end,
page_revision_start/end, section_path, section_id, heading_text,
clause_path, parent_id, child_id, child_order, content_type,
continues_from/to, table_id, table_header, code_or_defined_term,
cross_references, extraction_method, parser_version
```

`content_type` should distinguish `definition`, `provision`, `table`, `code_list`, `schedule`, `form`, `front_matter`, and `navigation`. Extract canonical cross-references such as `A.3(B)(xxii)` and `I.3(B)` into both metadata and optional link edges; they are frequent and material to legal answers.

### Retrieval

- Embed and rank **children** only. Retrieve 8–12 candidates, diversify by `parent_id`/section, then use a lightweight lexical boost or reranker for exact identifiers, acronyms, amounts, and BOP codes. Return the best 4–6 after diversification; the existing five-result context budget is compatible with the proposed child size.
- For each selected child, expand the LLM context with its short parent heading/breadcrumb and immediately adjacent child only when `continues_from/to` is set or the query targets a sequence/list. Do not indiscriminately append whole parents; that would dilute evidence and consume the strict `ChatService._select_within_budget` context prefix.
- Render citations as section/clause plus printed PDF page, e.g. `B.4(A)(i), p. 98`, rather than filename/chunk number alone. Treat a K-section cross-reference as a retrieval expansion candidate, not proof: retrieve the target provision before answering.

### Is the existing splitter sufficient?

**No for this manual.** It is useful as the final fallback inside an already identified overlong provision, and its row-boundary/header-repeat behavior is worth retaining. It is not sufficient as the primary splitter because it neither recognizes the manual’s heading/clause hierarchy nor persists source location/relationship metadata. Merely increasing `CHUNK_SIZE` would reduce fragmentation but would still mix independent rules and make exceptions harder to retrieve.

### Concrete work, ranked by priority (recommendations only; no implementation performed)

1. **P0 — Page-aware extraction contract:** extend the PDF parser output internally to retain page number, page revision, block coordinates, and table boundaries; remove repeated furniture by coordinates; add regression fixtures from pp. 15, 24, 47–49, 98–101, 198–201, and 259–261.
2. **P1 — Legal hierarchy parser and clause-first chunker:** add heading/clause detection, multi-page continuation, 900/1,400-character child rules, forced-split-only overlap, table/code-list/form handlers, and breadcrumb injection. Retain the current recursive splitter only as the fallback.
3. **P1 — Storage/citation metadata:** migrate the chunk schema and search result model for parent/child IDs, section/clause paths, page ranges, content types, and cross-reference links; ensure re-ingestion replaces old flat chunks.
4. **P2 — Parent/child retrieval:** add candidate diversification, exact identifier/amount/code matching, conditional adjacent-child expansion, and legal-location citations; revise chat context formatting to expose those citations.
5. **P2 — Evaluation corpus:** build document-specific questions covering definitions, monetary limits, nested exceptions, cross-page provisions, table membership, BOP-code lookup, and K-section cross-references. Compare flat 500/100 chunks against the new strategy for citation accuracy and completeness before changing defaults globally.

## Inspection limits

This assessment used pdfplumber’s text layer, page geometry on a representative content page, `find_tables()` detection, and read-only execution of the repository parser/chunker. I did not perform image-by-image visual QA of all 298 pages, OCR quality testing for scanned inserts, legal interpretation, live Ollama retrieval evaluation, or implementation. `find_tables()` missed some visually structured code lists, so the proposed parser must be validated on representative rendered pages rather than treating table detection as complete.
