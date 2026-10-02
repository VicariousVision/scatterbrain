"""Clause-first hierarchy, table, definition, and overlap regressions."""

from models.content import ExtractedBlock, SourceSpan
from services.legal_chunker import chunk_parsed_document
from services.text_cleaner import clean_parsed_document
from backend.tests.fixtures.legal_manual_snippets import manual_document
from config import settings


def _children(document):
    return [record for record in chunk_parsed_document(document) if record.record_type == "child"]


def test_nested_clause_cross_page_exception_and_stable_ids() -> None:
    document = manual_document(
        "B.4  Single discretionary allowance\n(A)  Allowance per calendar year\n    (i) Residents aged 18 or older may transfer R2 million.\n        (a) Documentary proof is required above that amount.\n        provided that the Authorised Dealer verifies the proof under I.3(B).",
        "        The verification must be retained for five years.\n    (ii) Minors receive a lower allowance.",
    )
    first = _children(document)
    repeated = _children(document)
    clause = next(record for record in first if record.clause_path == "B.4(A)(i)")
    assert clause.pdf_page_start == 1 and clause.pdf_page_end == 2
    assert "provided that" in clause.source_text
    assert "retained for five years" in clause.source_text
    assert clause.cross_references == ["I.3(B)"]
    assert "B.4" in clause.embedding_text and clause.embedding_text != clause.source_text
    assert [record.child_id for record in first] == [record.child_id for record in repeated]
    other = next(record for record in first if record.clause_path == "B.4(A)(ii)")
    assert "Documentary proof" not in other.source_text


def test_leading_marker_range_reference_is_not_a_new_clause() -> None:
    document = manual_document(
        "B.2  Rule\n(A)  Subsection\n(i) First rule.\n(ii) Second rule.\n"
        "(i) and (ii) above are subject to appropriate tax treatment.\n(iii) Third rule."
    )
    children = _children(document)
    assert [record.clause_path for record in children].count("B.2(A)(i)") == 1
    second = next(record for record in children if record.clause_path == "B.2(A)(ii)")
    assert "subject to appropriate tax treatment" in second.source_text


def test_indented_canonical_cross_reference_is_not_a_section_heading() -> None:
    document = manual_document(
        "B.2  Rule\n(A)  Subsection\n(i) Main obligation.\n"
        "                    I.3(C) of the Authorised Dealer Manual applies.\n"
        "(ii) Next obligation."
    )
    children = _children(document)
    assert [record.clause_path for record in children] == ["B.2(A)(i)", "B.2(A)(ii)"]
    assert "I.3(C)" in children[0].cross_references


def test_running_section_change_resets_hierarchy_after_navigation_gap() -> None:
    document = manual_document(
        "F.2  Trade\n(A) Exports\n(i) Export rule.",
        "(A) Securities\n(i) Securities rule.",
    )
    page = document.pages[1]
    page.section_marker = "G."
    for block in page.blocks:
        block.section_marker = "G."
    children = _children(document)
    assert {record.clause_path for record in children} == {"F.2(A)(i)", "G.(A)(i)"}


def test_separate_code_lists_have_unique_stable_ids() -> None:
    document = manual_document(
        "J.  Codes\n511 01 First code\n511 02 Second code\n"
        "(A) Another list\n600 01 Third code\n600 02 Fourth code"
    )
    records = chunk_parsed_document(document)
    code_records = [record for record in records if record.content_type == "code_list"]
    assert len({record.id for record in code_records}) == len(code_records)


def test_definitions_keep_nested_conditions_and_have_distinct_ids() -> None:
    document = manual_document(
        "A.1  Definitions\nAffected person means a person who:\n    (i) controls the entity; and\n    (ii) owns its voting rights.\nADLA means an Authorised Dealer with limited authority."
    )
    children = _children(document)
    definitions = {record.defined_term: record for record in children if record.content_type == "definition"}
    assert "(ii) owns its voting rights" in definitions["Affected person"].source_text
    assert definitions["Affected person"].child_id != definitions["ADLA"].child_id


def test_forced_split_alone_gets_exact_continuity_tail(monkeypatch) -> None:
    monkeypatch.setattr(settings, "legal_chunk_target_chars", 120)
    monkeypatch.setattr(settings, "legal_chunk_hard_max_chars", 180)
    monkeypatch.setattr(settings, "legal_forced_split_overlap_chars", 20)
    long_rule = " ".join(f"word{i}" for i in range(100))
    document = manual_document(
        f"B.4  Rule\n(A)  Allowance\n(i) {long_rule}\n(ii) Independent clause marker unique-two."
    )
    children = _children(document)
    forced = [record for record in children if record.clause_path == "B.4(A)(i)"]
    assert len(forced) > 1
    assert all(len(record.source_text) <= 180 for record in children)
    for previous, current in zip(forced, forced[1:]):
        assert current.source_text[:20] == previous.source_text[-20:]
        assert previous.continues_to == current.child_id
        assert current.continues_from == previous.child_id
    independent = next(record for record in children if record.clause_path == "B.4(A)(ii)")
    assert forced[-1].source_text[-20:] not in independent.source_text


def test_adjacent_page_tables_stitch_and_repeat_true_header(monkeypatch) -> None:
    monkeypatch.setattr(settings, "legal_chunk_target_chars", 60)
    monkeypatch.setattr(settings, "legal_chunk_hard_max_chars", 140)
    monkeypatch.setattr(settings, "legal_forced_split_overlap_chars", 20)
    document = manual_document("B.4 Rule", "continuation")
    for page, code in zip(document.pages, ("511 01", "511 02")):
        page.blocks.append(
            ExtractedBlock(
                block_type="table",
                raw_text=f"| Code | Description |\n| --- | --- |\n| {code} | Meaning {code} |",
                clean_text=f"| Code | Description |\n| --- | --- |\n| {code} | Meaning {code} |",
                order=1,
                pdf_page=page.pdf_page,
                printed_page=page.printed_page,
                page_revision=page.page_revision,
                section_marker="J.",
                table_id=f"t{page.pdf_page}",
                table_header=["Code", "Description"],
                table_rows=[[code, f"Meaning {code}"]],
                source_spans=[SourceSpan(pdf_page=page.pdf_page, block_order=1)],
            )
        )
    records = chunk_parsed_document(document)
    table_parents = [r for r in records if r.record_type == "parent" and r.content_type == "table"]
    table_children = [r for r in records if r.record_type == "child" and r.content_type == "table"]
    assert len(table_parents) == 1
    assert table_parents[0].pdf_page_start == 1 and table_parents[0].pdf_page_end == 2
    assert all(child.source_text.startswith("| Code | Description |") for child in table_children)
    assert {child.pdf_page_start for child in table_children} == {1, 2}
    assert all(child.pdf_page_start == child.pdf_page_end for child in table_children)
    assert "511 01" in table_parents[0].source_text and "511 02" in table_parents[0].source_text


def test_bop_codes_stay_with_descriptions_and_generic_fallback(monkeypatch) -> None:
    document = manual_document(
        "J.  FinSurv Reporting System\n511 01    Investment allowance - shares\n511 02    Investment allowance - property"
    )
    code_children = [r for r in _children(document) if r.content_type == "code_list"]
    assert code_children
    assert "511 01" in (code_children[0].code or "")
    assert "511 01  Investment allowance - shares" in code_children[0].source_text

    generic = manual_document("ordinary unrelated prose")
    generic.is_legal = False
    generic.document_title = None
    generic.source_filename = "notes.txt"
    monkeypatch.setattr(settings, "chunk_size", 20)
    monkeypatch.setattr(settings, "chunk_overlap", 5)
    records = chunk_parsed_document(generic)
    assert records and all(r.content_type == "generic" for r in records)


def test_oversized_parent_creates_intermediate_parent(monkeypatch) -> None:
    monkeypatch.setattr(settings, "legal_parent_max_chars", 100)
    document = manual_document(
        "B.4  Rule\n(A)  Large subsection\n"
        "(i) First governing provision with enough explanatory language to exceed the parent threshold.\n"
        "(ii) Second governing provision with additional explanatory language and conditions."
    )
    records = chunk_parsed_document(document)
    intermediate = [r for r in records if r.record_type == "intermediate_parent"]
    assert intermediate
    child_parent_ids = {r.parent_id for r in records if r.record_type == "child"}
    assert {r.parent_id for r in intermediate} <= child_parent_ids


def test_long_code_description_is_split_without_losing_code(monkeypatch) -> None:
    monkeypatch.setattr(settings, "legal_chunk_hard_max_chars", 180)
    monkeypatch.setattr(settings, "legal_forced_split_overlap_chars", 20)
    description = " ".join(f"description{i}" for i in range(80))
    document = manual_document(
        f"J.  FinSurv Reporting System\n511 01    {description}\n511 02    Short independent description"
    )
    records = chunk_parsed_document(document)
    parent = next(r for r in records if r.record_type == "parent" and r.content_type == "code_list")
    children = [r for r in records if r.record_type == "child" and r.content_type == "code_list"]
    assert description in parent.source_text
    assert all(len(child.source_text) <= 180 for child in children)
    assert all(child.source_text.startswith(("511 01", "511 02")) for child in children)


def test_wrapped_heading_and_governing_preamble_are_retained() -> None:
    document = manual_document(
        "B.4  Single discretionary allowance and other miscellaneous payments for\n"
        "private individuals\n"
        "The following rules govern resident transfers:\n"
        "(A)  Annual allowance\n"
        "(i) Residents may transfer the prescribed amount."
    )
    records = chunk_parsed_document(document)
    parent = next(
        record
        for record in records
        if record.record_type == "parent" and record.clause_path == "B.4(A)"
    )
    child = next(
        record
        for record in records
        if record.record_type == "child" and record.clause_path == "B.4(A)(i)"
    )
    assert "The following rules govern resident transfers" in parent.source_text
    assert "private individuals" in child.breadcrumb


def test_external_form_and_schedule_references_remain_provisions() -> None:
    document = manual_document(
        "C.  Gold exports\n"
        "(B) Other exports of gold\n"
        "(i) Gold in any form must be referred to the regulator using the "
        "external specimen described in Schedule 2."
    )
    records = chunk_parsed_document(document)
    relevant = [
        record
        for record in records
        if record.record_type in {"parent", "child"}
    ]
    assert relevant
    assert {record.content_type for record in relevant} == {"provision"}


def test_description_row_continues_across_pages_with_complete_provenance() -> None:
    document = manual_document("K.  Returns and reports", "continuation")
    rows_by_page = (
        [["Local outsourcing", "", "", "A service within South Africa but", "", ""]],
        [
            ["", "", "", "not based at the reporting entity premises.", "", ""],
            ["International outsourcing", "", "", "A service outside South Africa.", "", ""],
        ],
    )
    for page, rows in zip(document.pages, rows_by_page):
        raw_rows = "\n".join("| " + " | ".join(row) + " |" for row in rows)
        page.blocks.append(
            ExtractedBlock(
                block_type="table",
                raw_text=(
                    "|  | Concept |  |  | Description |  |\n"
                    "| --- | --- | --- | --- | --- | --- |\n"
                    f"{raw_rows}"
                ),
                clean_text=(
                    "|  | Concept |  |  | Description |  |\n"
                    "| --- | --- | --- | --- | --- | --- |\n"
                    f"{raw_rows}"
                ),
                order=1,
                pdf_page=page.pdf_page,
                printed_page=page.printed_page,
                page_revision=page.page_revision,
                section_marker="K.",
                table_id=f"concept-{page.pdf_page}",
                table_header=["", "Concept", "", "", "Description", ""],
                table_rows=rows,
                source_spans=[SourceSpan(pdf_page=page.pdf_page, block_order=1)],
            )
        )

    records = chunk_parsed_document(document)
    parent = next(
        record
        for record in records
        if record.record_type == "parent" and record.content_type == "table"
    )
    children = [
        record
        for record in records
        if record.record_type == "child" and record.content_type == "table"
    ]
    assert "within South Africa but not based" in parent.source_text
    assert all(child.source_text.startswith("|  | Concept |") for child in children)
    local = next(child for child in children if "Local outsourcing" in child.source_text)
    assert (local.pdf_page_start, local.pdf_page_end) == (1, 2)
    assert "within South Africa but" in local.raw_text
    assert "not based at the reporting entity" in local.raw_text


def test_true_table_header_and_unambiguous_entity_key_are_repeated() -> None:
    document = manual_document("G.  Securities control")
    page = document.pages[0]
    rows = [
        ["", "", "as Authorised Bank", ""],
        ["ABSA Bank", "Cape Town", "", ""],
        ["", "Durban", "", ""],
    ]
    page.blocks.append(
        ExtractedBlock(
            block_type="table",
            raw_text="table source",
            clean_text="table source",
            order=1,
            pdf_page=1,
            printed_page=page.printed_page,
            section_marker="G.",
            table_id="banks",
            table_header=["Authorised Dealer", "", "Branch or entity appointed", ""],
            table_rows=rows,
            source_spans=[SourceSpan(pdf_page=1, block_order=1)],
        )
    )
    records = chunk_parsed_document(document)
    parent = next(
        record
        for record in records
        if record.record_type == "parent" and record.content_type == "table"
    )
    assert "Branch or entity appointed as Authorised Bank" in parent.source_text
    assert "| ABSA Bank | Durban |" in parent.source_text


def test_forced_child_raw_text_remains_extracted_source(monkeypatch) -> None:
    monkeypatch.setattr(settings, "legal_chunk_target_chars", 120)
    monkeypatch.setattr(settings, "legal_chunk_hard_max_chars", 180)
    monkeypatch.setattr(settings, "legal_forced_split_overlap_chars", 20)
    raw_rule = "  ".join(f"word{index}" for index in range(100))
    document = clean_parsed_document(
        manual_document(f"B.4  Rule\n(A) Allowance\n(i) {raw_rule}")
    )
    forced = [
        record
        for record in _children(document)
        if record.clause_path == "B.4(A)(i)"
    ]
    assert len(forced) > 1
    assert all(len(record.source_text) <= 180 for record in forced)
    assert all("  " in record.raw_text for record in forced)
    assert any(record.raw_text != record.source_text for record in forced)
