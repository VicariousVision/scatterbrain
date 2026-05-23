"""
Property-based tests for the entity extractor service.

**Validates: Requirements 3.2, 3.3, 3.4, 3.6**
"""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from backend.models.entities import Entity, ExtractionResult, Relationship
from backend.services.entity_extractor import (
    _build_extraction_result,
    _parse_llm_response,
    build_extraction_prompt,
)


# ---------------------------------------------------------------------------
# Shared strategy: arbitrary text strings (including empty)
# ---------------------------------------------------------------------------

_any_text = st.text()


# ---------------------------------------------------------------------------
# Property 4: Extraction prompt always contains all required entity types
# ---------------------------------------------------------------------------

_REQUIRED_ENTITY_TYPES = [
    "Person",
    "Organization",
    "Contract",
    "Clause",
    "Date",
    "Jurisdiction",
    "Obligation",
]


@given(text=_any_text)
@settings(max_examples=50)
def test_prompt_contains_all_entity_types(text: str) -> None:
    """
    Property 4: Extraction prompt always contains all required entity types.

    For any input document text, the prompt constructed by build_extraction_prompt
    must contain all 7 legal entity type names: Person, Organization, Contract,
    Clause, Date, Jurisdiction, and Obligation.

    **Validates: Requirements 3.2**
    """
    prompt = build_extraction_prompt(text)

    for entity_type in _REQUIRED_ENTITY_TYPES:
        assert entity_type in prompt, (
            f"Expected entity type '{entity_type}' to appear in the extraction "
            f"prompt, but it was not found."
        )


# ---------------------------------------------------------------------------
# Property 5: Extraction prompt always specifies relationship schema
# ---------------------------------------------------------------------------

_REQUIRED_RELATIONSHIP_FIELDS = [
    "source_entity",
    "relationship_type",
    "target_entity",
]


@given(text=_any_text)
@settings(max_examples=50)
def test_prompt_specifies_relationship_schema(text: str) -> None:
    """
    Property 5: Extraction prompt always specifies relationship schema.

    For any input document text, the prompt constructed by build_extraction_prompt
    must contain the three relationship field names: source_entity,
    relationship_type, and target_entity.

    **Validates: Requirements 3.3**
    """
    prompt = build_extraction_prompt(text)

    for field in _REQUIRED_RELATIONSHIP_FIELDS:
        assert field in prompt, (
            f"Expected relationship field '{field}' to appear in the extraction "
            f"prompt, but it was not found."
        )


# ---------------------------------------------------------------------------
# Strategies for Properties 6 and 7
# ---------------------------------------------------------------------------

# Non-empty printable text for entity/relationship field values.
_nonempty_text = st.text(min_size=1)

# Strategy for a single entity dict as the LLM would return it.
_entity_dict_strategy = st.fixed_dictionaries(
    {
        "name": _nonempty_text,
        "type": st.sampled_from(
            ["Person", "Organization", "Contract", "Clause", "Date", "Jurisdiction", "Obligation"]
        ),
    }
)

# Strategy for a single relationship dict as the LLM would return it.
_relationship_dict_strategy = st.fixed_dictionaries(
    {
        "source_entity": _nonempty_text,
        "relationship_type": _nonempty_text,
        "target_entity": _nonempty_text,
    }
)

# Strategy for the full extraction payload dict (already parsed from JSON).
_extraction_data_strategy = st.fixed_dictionaries(
    {
        "entities": st.lists(_entity_dict_strategy, min_size=1, max_size=10),
        "relationships": st.lists(_relationship_dict_strategy, min_size=1, max_size=10),
    }
)

# Strategy for a document_id: non-empty printable ASCII string.
_document_id_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
    min_size=1,
    max_size=36,
)


# ---------------------------------------------------------------------------
# Property 6: Valid LLM JSON response parses to correct object structure
# ---------------------------------------------------------------------------


@given(data=_extraction_data_strategy)
@settings(max_examples=50)
def test_valid_json_parses_to_correct_structure(data: dict) -> None:
    """
    Property 6: Valid LLM JSON response parses to correct object structure.

    For any valid JSON string conforming to the extraction schema, parsing it
    with _parse_llm_response and then building the result with
    _build_extraction_result must produce an ExtractionResult whose entities
    and relationships lists are non-empty and where every field on every object
    is non-null.

    **Validates: Requirements 3.4**
    """
    raw_json = json.dumps(data)
    document_id = "test-doc-id"

    parsed = _parse_llm_response(raw_json)
    result = _build_extraction_result(parsed, document_id)

    assert isinstance(result, ExtractionResult)

    # Both lists must be populated (we generated min_size=1 for each).
    assert len(result.entities) > 0, "Expected at least one entity in the result"
    assert len(result.relationships) > 0, "Expected at least one relationship in the result"

    # Every entity field must be non-null and non-empty.
    for entity in result.entities:
        assert isinstance(entity, Entity)
        assert entity.id, "entity.id must be non-empty"
        assert entity.name, "entity.name must be non-empty"
        assert entity.type, "entity.type must be non-empty"
        assert entity.document_id, "entity.document_id must be non-empty"

    # Every relationship field must be non-null and non-empty.
    for rel in result.relationships:
        assert isinstance(rel, Relationship)
        assert rel.source_entity, "relationship.source_entity must be non-empty"
        assert rel.relationship_type, "relationship.relationship_type must be non-empty"
        assert rel.target_entity, "relationship.target_entity must be non-empty"
        assert rel.document_id, "relationship.document_id must be non-empty"


# ---------------------------------------------------------------------------
# Property 7: All extracted entities and relationships carry the document_id
# ---------------------------------------------------------------------------


@given(data=_extraction_data_strategy, document_id=_document_id_strategy)
@settings(max_examples=50)
def test_document_id_propagated_to_all_objects(data: dict, document_id: str) -> None:
    """
    Property 7: All extracted entities and relationships carry the document_id.

    For any extraction result produced for a given document_id, every entity
    object and every relationship object in the result must have its
    document_id field set to that exact same document_id.

    **Validates: Requirements 3.6**
    """
    result = _build_extraction_result(data, document_id)

    assert isinstance(result, ExtractionResult)

    for entity in result.entities:
        assert entity.document_id == document_id, (
            f"Expected entity.document_id == {document_id!r}, "
            f"got {entity.document_id!r}"
        )

    for rel in result.relationships:
        assert rel.document_id == document_id, (
            f"Expected relationship.document_id == {document_id!r}, "
            f"got {rel.document_id!r}"
        )
