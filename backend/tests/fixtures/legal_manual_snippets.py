"""Compact synthetic legal-manual fixtures; no copyrighted PDF pages."""

from __future__ import annotations

import hashlib
import re

from models.content import ExtractedBlock, ExtractedPage, ParsedDocument, SourceSpan

_SECTION_RE = re.compile(r"(?m)^\s*((?:[A-K](?:\.\d+)+)|(?:[A-K]\.))\s+")


def manual_document(*page_texts: str) -> ParsedDocument:
    pages = []
    current_section = "B.4"
    for index, text in enumerate(page_texts, start=1):
        match = _SECTION_RE.search(text)
        if match:
            current_section = match.group(1).rstrip(".")
            if "." not in current_section:
                current_section += "."
        span = SourceSpan(
            pdf_page=index,
            printed_page=str(97 + index),
            page_revision="6/2026",
            bbox=(40, 80, 550, 760),
            block_order=0,
        )
        pages.append(
            ExtractedPage(
                pdf_page=index,
                printed_page=str(97 + index),
                page_revision="6/2026",
                section_marker=current_section,
                blocks=[
                    ExtractedBlock(
                        raw_text=text,
                        clean_text=text,
                        order=0,
                        bbox=(40, 80, 550, 760),
                        pdf_page=index,
                        printed_page=str(97 + index),
                        page_revision="6/2026",
                        section_marker=current_section,
                        source_spans=[span],
                        extraction_method="fixture",
                    )
                ],
            )
        )
    data = "\n".join(page_texts).encode()
    return ParsedDocument(
        source_filename="manual.pdf",
        source_sha256=hashlib.sha256(data).hexdigest(),
        document_title="Currency and Exchanges Manual for Authorised Dealers",
        document_version="fixture",
        extraction_method="fixture",
        pages=pages,
        is_legal=True,
    )
