"""Entity and relationship extractor using the Ollama LLM.

Sends document text to the LLM with a structured prompt and parses the JSON
response into ``ExtractionResult`` objects containing ``Entity`` and
``Relationship`` instances.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

from __future__ import annotations

import json
import uuid

from backend.models.entities import Entity, ExtractionResult, Relationship
from backend.services.ollama_client import OllamaClient

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class ExtractionError(Exception):
    """Raised when the LLM fails to return valid JSON after all retry attempts."""


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT_TEMPLATE = """\
You are a legal document analysis assistant. Extract all entities and relationships from the following legal document text.

Return ONLY a valid JSON object with this exact structure:
{{
  "entities": [
    {{"name": "string", "type": "Person|Organization|Contract|Clause|Date|Jurisdiction|Obligation"}}
  ],
  "relationships": [
    {{"source_entity": "string", "relationship_type": "string", "target_entity": "string"}}
  ]
}}

Entity types to identify: Person, Organization, Contract, Clause, Date, Jurisdiction, Obligation
Each relationship must have a source_entity, relationship_type, and target_entity.
Do not include any text outside the JSON object.

Document text:
{document_text}"""

# Maximum number of total attempts (1 initial + 2 retries = 3 total).
_MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_extraction_prompt(text: str) -> str:
    """Return the extraction prompt for the given document text.

    Exposed as a standalone function so that property tests can verify the
    prompt structure without needing a live Ollama instance.

    Args:
        text: Raw document text to embed in the prompt.

    Returns:
        The fully-rendered prompt string.
    """
    return _EXTRACTION_PROMPT_TEMPLATE.format(document_text=text)


async def extract_entities(
    text: str,
    document_id: str,
    client: OllamaClient,
) -> ExtractionResult:
    """Extract entities and relationships from document text via the LLM.

    Sends ``text`` to the LLM using ``client``, parses the JSON response, and
    returns an ``ExtractionResult`` with every ``Entity`` and ``Relationship``
    stamped with ``document_id``.

    Retries up to ``_MAX_ATTEMPTS`` times (3 total) on malformed or non-JSON
    responses before raising ``ExtractionError``.

    Args:
        text:        Raw document text to analyse.
        document_id: Identifier of the source document; attached to every
                     extracted entity and relationship.
        client:      An ``OllamaClient`` instance used to call the LLM.

    Returns:
        An ``ExtractionResult`` containing populated entity and relationship
        lists, each carrying the given ``document_id``.

    Raises:
        ExtractionError: If the LLM returns malformed JSON on all attempts.
    """
    prompt = build_extraction_prompt(text)
    last_error: Exception | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        raw_response = await client.generate(prompt)

        try:
            data = _parse_llm_response(raw_response)
            return _build_extraction_result(data, document_id)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            last_error = exc
            # Log the failure and retry (unless this was the last attempt).
            if attempt < _MAX_ATTEMPTS:
                continue

    raise ExtractionError(
        f"Entity extraction failed after {_MAX_ATTEMPTS} attempts. "
        f"Last error: {last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_llm_response(raw: str) -> dict:
    """Parse the raw LLM output as JSON, stripping any surrounding whitespace.

    The LLM is instructed to return only a JSON object, but may occasionally
    wrap it in markdown code fences.  This helper strips common wrappers
    before attempting to parse.

    Args:
        raw: The raw string returned by the LLM.

    Returns:
        A dict with ``"entities"`` and ``"relationships"`` keys.

    Raises:
        json.JSONDecodeError: If the string cannot be parsed as JSON.
        ValueError:           If the parsed object is missing required keys.
    """
    text = raw.strip()

    # Strip optional markdown code fences (```json ... ``` or ``` ... ```).
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove opening fence line and closing fence line.
        inner_lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner_lines).strip()

    data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
    if "entities" not in data or "relationships" not in data:
        raise ValueError(
            "JSON response is missing required keys 'entities' and/or 'relationships'"
        )

    return data


def _build_extraction_result(data: dict, document_id: str) -> ExtractionResult:
    """Convert the parsed LLM JSON dict into an ``ExtractionResult``.

    Generates a UUID4 ``id`` for each entity and stamps every entity and
    relationship with ``document_id``.

    Args:
        data:        Parsed dict with ``"entities"`` and ``"relationships"`` lists.
        document_id: The document identifier to attach to every object.

    Returns:
        A fully-populated ``ExtractionResult``.
    """
    entities: list[Entity] = []
    for raw_entity in data["entities"]:
        entities.append(
            Entity(
                id=str(uuid.uuid4()),
                name=raw_entity["name"],
                type=raw_entity["type"],
                document_id=document_id,
            )
        )

    relationships: list[Relationship] = []
    for raw_rel in data["relationships"]:
        relationships.append(
            Relationship(
                source_entity=raw_rel["source_entity"],
                relationship_type=raw_rel["relationship_type"],
                target_entity=raw_rel["target_entity"],
                document_id=document_id,
            )
        )

    return ExtractionResult(entities=entities, relationships=relationships)
