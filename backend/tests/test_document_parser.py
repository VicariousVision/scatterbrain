"""Page-aware parser regression tests using compact geometry fixtures."""

import json
from pathlib import Path

from services.document_parser import parse_document, parse_document_structured

_FIXTURE = Path(__file__).parent / "fixtures" / "manual_page_blocks.json"


def test_furniture_metadata_and_mixed_block_order_are_preserved() -> None:
    pages = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    document = parse_document_structured("manual.pdf", b"fixture", pdf_pages=pages)

    assert [(page.pdf_page, page.printed_page, page.page_revision) for page in document.pages] == [
        (98, "98", "6/2026"),
        (99, "99", "8/2026"),
    ]
    assert all(page.section_marker == "B.4" for page in document.pages)
    assert document.document_title == "Currency and Exchanges Manual for Authorised Dealers"
    assert "Currency and Exchanges Manual" not in document.flatten()
    assert "of 298" not in document.flatten()
    second = document.pages[1]
    assert [block.block_type for block in second.blocks] == ["prose", "table", "prose"]
    assert [block.raw_text for block in second.blocks][::2] == [
        "Prose before table",
        "Prose after table",
    ]
    assert second.blocks[1].source_spans[0].pdf_page == 99


def test_body_table_near_header_band_cannot_override_section_marker() -> None:
    pages = [
        {
            "pdf_page": 236,
            "blocks": [
                {
                    "text": "Currency and Exchanges Manual for Authorised Dealers H.",
                    "bbox": [70, 44, 525, 54],
                },
                {
                    "block_type": "table",
                    "text": "| Details |\n| --- |\n| Listed on a South African exchange. |",
                    "bbox": [119, 71.6, 535, 693],
                    "table_header": ["Details"],
                    "table_rows": [["Listed on a South African exchange."]],
                },
            ],
        }
    ]
    document = parse_document_structured("manual.pdf", b"near-band", pdf_pages=pages)
    page = document.pages[0]
    assert page.section_marker == "H."
    assert [block.block_type for block in page.blocks] == ["table"]


def test_version_control_and_toc_are_navigation_not_body() -> None:
    pages = [
        {
            "pdf_page": 2,
            "blocks": [{"text": "Version control sheet\nVersion number Issue date Circular number\n1.1 2024-01-01 1-2024", "bbox": [40, 90, 550, 300]}],
        },
        {
            "pdf_page": 5,
            "blocks": [{"text": "Table of Contents\nA.1 Definitions ................................ 15\nB.4 Allowance ................................ 98", "bbox": [40, 90, 550, 300]}],
        },
    ]
    document = parse_document_structured("manual.pdf", b"front", pdf_pages=pages)
    assert [page.content_type for page in document.pages] == ["front_matter", "navigation"]
    assert document.document_version == "1.1 (2024-01-01)"
    assert {entry["section_id"] for entry in document.navigation} == {"A.1", "B.4"}


def test_section_index_pages_are_navigation_until_body_restarts() -> None:
    pages = [
        {
            "pdf_page": 192,
            "section_marker": "G.",
            "blocks": [{"text": "G. Securities control\nIndex\n(A) Control over securities\n(i) Regulations", "bbox": [40, 90, 550, 500]}],
        },
        {
            "pdf_page": 193,
            "section_marker": "G.",
            "blocks": [{"text": "(B) Listing requirements\n(i) Capitalisation issues", "bbox": [40, 90, 550, 500]}],
        },
        {
            "pdf_page": 195,
            "section_marker": "G.",
            "blocks": [{"text": "(A) Control over securities\n(i) Regulations\n(a) Authorised Dealers must inspect proof.", "bbox": [40, 90, 550, 500]}],
        },
    ]
    document = parse_document_structured("manual.pdf", b"index", pdf_pages=pages)
    assert [page.content_type for page in document.pages] == ["navigation", "navigation", "body"]
    assert any(entry.get("clause_path") == "G.(A)(i)" for entry in document.navigation)


def test_txt_compatibility_wrapper_and_structured_page() -> None:
    content = "ordinary text".encode()
    assert parse_document("notes.txt", content) == "ordinary text"
    structured = parse_document_structured("notes.txt", content)
    assert structured.pages[0].pdf_page == 1
    assert structured.pages[0].blocks[0].raw_text == "ordinary text"
