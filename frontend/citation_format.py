"""Pure formatting helpers for additive backend citation payloads."""

from __future__ import annotations

from typing import Any, Iterable


def format_citation_label(citation: dict[str, Any]) -> str:
    """Return a compact legal/page label with safe legacy fallbacks."""
    supplied = str(citation.get("label") or "").strip()
    if supplied:
        return supplied
    location = str(
        citation.get("clause_path") or citation.get("section_id") or ""
    ).strip()
    printed_start = citation.get("printed_page_start")
    printed_end = citation.get("printed_page_end")
    pdf_start = citation.get("pdf_page_start")
    pdf_end = citation.get("pdf_page_end")
    page_start = printed_start if printed_start is not None else pdf_start
    page_end = printed_end if printed_end is not None else pdf_end
    page_label = ""
    if page_start is not None:
        if page_end is not None and str(page_end) != str(page_start):
            page_label = f"pp. {page_start}–{page_end}"
        else:
            page_label = f"p. {page_start}"
        if printed_start is not None and pdf_start is not None and str(printed_start) != str(pdf_start):
            pdf_range = str(pdf_start)
            if pdf_end is not None and str(pdf_end) != str(pdf_start):
                pdf_range = f"{pdf_start}–{pdf_end}"
            page_label += f" (PDF {pdf_range})"
    label = ", ".join(part for part in (location, page_label) if part)
    if not label:
        label = str(citation.get("filename") or "Unknown source")
    revisions = [str(value) for value in citation.get("page_revisions") or [] if value]
    if revisions:
        label += f" · rev. {'–'.join(dict.fromkeys(revisions))}"
    return label


def unique_citation_labels(citations: Iterable[dict[str, Any]]) -> list[str]:
    """Format and de-duplicate citations while preserving API order."""
    return list(dict.fromkeys(format_citation_label(item) for item in citations))
