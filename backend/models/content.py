"""Structured content models shared by ingestion, storage, and retrieval.

The models deliberately keep extracted source text separate from embedding
context. Breadcrumbs improve retrieval, but are never inserted into quoted
source text or citations.
"""

from __future__ import annotations

from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PARSER_VERSION = "legal-structured-v1"
SCHEMA_VERSION = 2


class SourceSpan(BaseModel):
    """Location of extracted source content.

    Parameters
    ----------
    pdf_page:
        One-based PDF page number.
    printed_page:
        Printed page label when detected; kept as text because legal manuals
        may use Roman numerals or prefixed page labels.
    page_revision:
        Revision/issue marker printed on the page.
    bbox:
        ``(x0, top, x1, bottom)`` coordinates in PDF points.
    block_order:
        Reading-order index on the page.
    char_start, char_end:
        Optional offsets within the block's raw text.
    """

    pdf_page: int = Field(ge=1)
    printed_page: str | None = None
    page_revision: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    block_order: int = Field(default=0, ge=0)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> "SourceSpan":
        """Reject a reversed character range."""
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end < self.char_start
        ):
            raise ValueError("char_end must be greater than or equal to char_start")
        return self


class ExtractedBlock(BaseModel):
    """Coordinate-aware prose or table block from one source page.

    Parameters
    ----------
    block_type:
        ``prose`` for positioned lines/paragraphs or ``table`` for an
        extracted visual table.
    raw_text:
        Faithful extracted text before cleaning.
    clean_text:
        Conservatively normalized text used by structural parsing.
    order:
        Reading-order index after coordinate sorting.
    bbox:
        Source coordinates in PDF points.
    """

    model_config = ConfigDict(extra="ignore")

    block_type: Literal["prose", "table"] = "prose"
    raw_text: str
    clean_text: str = ""
    order: int = Field(default=0, ge=0)
    bbox: tuple[float, float, float, float] | None = None
    pdf_page: int = Field(ge=1)
    printed_page: str | None = None
    page_revision: str | None = None
    section_marker: str | None = None
    source_spans: list[SourceSpan] = Field(default_factory=list)
    table_id: str | None = None
    table_header: list[str] = Field(default_factory=list)
    table_rows: list[list[str]] = Field(default_factory=list)
    extraction_method: str = ""
    parser_version: str = PARSER_VERSION


class ExtractedPage(BaseModel):
    """One extracted page with ordered blocks and page provenance.

    Parameters
    ----------
    pdf_page:
        One-based PDF page number.
    blocks:
        Blocks in visual reading order; structural parsing never has to infer
        page boundaries from a flattened string.
    content_type:
        ``body``, ``front_matter``, or ``navigation``.
    """

    model_config = ConfigDict(extra="ignore")

    pdf_page: int = Field(ge=1)
    width: float | None = None
    height: float | None = None
    printed_page: str | None = None
    page_revision: str | None = None
    section_marker: str | None = None
    content_type: Literal["body", "front_matter", "navigation"] = "body"
    blocks: list[ExtractedBlock] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    """Page-aware parsed document passed to the structural chunker.

    Parameters
    ----------
    source_filename:
        Original uploaded filename.
    source_sha256:
        SHA-256 of the exact uploaded bytes.
    pages:
        Ordered extracted page objects.
    """

    model_config = ConfigDict(extra="ignore")

    source_filename: str
    source_sha256: str
    document_title: str | None = None
    document_version: str | None = None
    extraction_method: str
    parser_version: str = PARSER_VERSION
    schema_version: int = SCHEMA_VERSION
    pages: list[ExtractedPage] = Field(default_factory=list)
    navigation: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_legal: bool = False

    def flatten(self, *, include_navigation: bool = True) -> str:
        """Return a compatibility text view without losing stored page data."""
        parts: list[str] = []
        for page in self.pages:
            if not include_navigation and page.content_type != "body":
                continue
            block_texts: list[str] = []
            for block in page.blocks:
                value = (block.clean_text or block.raw_text).strip("\n")
                if value.strip():
                    block_texts.append(value)
            text = "\n\n".join(block_texts)
            if text:
                parts.append(text)
        return "\n\n".join(parts)


RecordType = Literal[
    "child",
    "parent",
    "intermediate_parent",
    "front_matter",
    "navigation",
]


class ChunkRecord(BaseModel):
    """Structured parent or searchable child record.

    Parent records preserve complete governing source but are never embedded.
    Only records with ``record_type='child'`` are inserted into ``vec_chunks``.

    Parameters
    ----------
    record_type:
        Storage/ranking role.
    source_text:
        Clean source quotation, without an injected breadcrumb.
    embedding_text:
        Search text, normally breadcrumb plus source text for children.
    parent_id, child_id:
        Stable hash-derived identifiers.
    """

    model_config = ConfigDict(extra="ignore")

    record_type: RecordType = "child"
    document_id: str | None = None
    document_title: str | None = None
    source_filename: str
    source_sha256: str
    document_version: str | None = None
    pdf_page_start: int | None = Field(default=None, ge=1)
    pdf_page_end: int | None = Field(default=None, ge=1)
    printed_page_start: str | None = None
    printed_page_end: str | None = None
    page_revisions: list[str] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    section_id: str | None = None
    clause_path: str | None = None
    heading: str | None = None
    breadcrumb: str = ""
    parent_id: str | None = None
    ancestor_parent_id: str | None = None
    child_id: str | None = None
    child_order: int = Field(default=0, ge=0)
    parent_order: int = Field(default=0, ge=0)
    content_type: Literal[
        "generic",
        "provision",
        "definition",
        "table",
        "code_list",
        "schedule",
        "form",
        "form_instructions",
        "form_field_group",
        "front_matter",
        "navigation",
    ] = "generic"
    continues_from: str | None = None
    continues_to: str | None = None
    table_id: str | None = None
    table_header: list[str] = Field(default_factory=list)
    code: str | None = None
    defined_term: str | None = None
    cross_references: list[str] = Field(default_factory=list)
    extraction_method: str = ""
    parser_version: str = PARSER_VERSION
    source_spans: list[SourceSpan] = Field(default_factory=list)
    raw_text: str = ""
    source_text: str = ""
    embedding_text: str = ""
    is_structured: bool = True

    @model_validator(mode="after")
    def validate_record_identity(self) -> "ChunkRecord":
        """Require the stable identifier appropriate for the record role."""
        if self.record_type == "child" and not self.child_id:
            raise ValueError("child records require child_id")
        if self.record_type != "child" and not self.parent_id:
            raise ValueError("parent/navigation records require parent_id")
        if (
            self.pdf_page_start is not None
            and self.pdf_page_end is not None
            and self.pdf_page_end < self.pdf_page_start
        ):
            raise ValueError("pdf_page_end must be >= pdf_page_start")
        return self

    @property
    def id(self) -> str:
        """Return the stable child or parent identifier."""
        return self.child_id or self.parent_id or ""


class Citation(BaseModel):
    """Additive legal/source citation returned by chat APIs.

    Parameters
    ----------
    filename:
        Original source filename; always available for legacy fallback.
    clause_path, section_id:
        Structured legal location when available.
    label:
        Preformatted compact display label.
    """

    document_id: str | None = None
    child_id: str | None = None
    filename: str
    section_id: str | None = None
    clause_path: str | None = None
    heading: str | None = None
    printed_page_start: str | None = None
    printed_page_end: str | None = None
    pdf_page_start: int | None = None
    pdf_page_end: int | None = None
    page_revisions: list[str] = Field(default_factory=list)
    label: str = ""


class RetrievedContext(BaseModel):
    """Ranked child plus optional governing/relationship context.

    Parameters
    ----------
    child_id:
        Stable searchable child identifier.
    source_text:
        Faithful cleaned quotation supplied to the LLM.
    metadata:
        Additive source, page, hierarchy, and relationship fields.
    """

    child_id: str
    source_text: str
    embedding_text: str = ""
    score: float = 0.0
    similarity: float = 0.0
    rank: int = 0
    relation: Literal["primary", "continuation", "adjacent", "cross_reference"] = (
        "primary"
    )
    governing_context: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResult(BaseModel):
    """Typed result from ``ChatService`` with tuple compatibility.

    Parameters
    ----------
    answer:
        Generated answer text.
    contexts:
        Exactly the source contexts presented to the model.
    citations:
        Deduplicated citations for those contexts.

    Iteration yields ``(answer, source_texts)`` so existing in-repository
    callers that unpacked the historical two-tuple continue to work.
    """

    answer: str
    contexts: list[RetrievedContext] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)

    @property
    def source_texts(self) -> list[str]:
        """Exactly the source texts presented to the LLM."""
        return [context.source_text for context in self.contexts]

    def __iter__(self) -> Iterator[Any]:
        yield self.answer
        yield self.source_texts

    def __eq__(self, other: object) -> bool:
        if isinstance(other, tuple) and len(other) == 2:
            return (self.answer, self.source_texts) == other
        return super().__eq__(other)
