"""Entity extraction and Neo4j MERGE loader for the SARB manual.

Step 4 of the ingestion pipeline:
  1. Takes each ProvisionChunk from provision_chunker.py.
  2. Sends it to Mistral 7B (Ollama, CPU) with a domain-specific JSON prompt.
  3. Parses the extraction result.
  4. MERGEs everything into Neo4j against the schema in graph_schema_service.py.

Concurrency
-----------
The OllamaClient already serialises calls via a module-level Semaphore(1).
At this layer we use asyncio.gather over the provision list so the event loop
can pipeline I/O (Neo4j writes, prompt construction) while Ollama is busy.
On CPU-only hardware with 24 GB RAM, OLLAMA_NUM_PARALLEL=1 is the right cap;
if you set it to 2 and test stability, raise _EXTRACTION_CONCURRENCY to 2.

Output shapes accepted from the LLM (see prompt below):
{
  "defined_terms_used": [str],
  "cross_references": [str],
  "limits": [{"amount": number, "currency": str, "period": str, "description": str}],
  "party_roles": [str],
  "legal_instruments": [str],
  "entities": [str]
}
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from neo4j import Driver
from neo4j.exceptions import Neo4jError

from backend.services.ollama_client import OllamaClient, OllamaClientError
from backend.services.provision_chunker import DefinitionChunk, ProvisionChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concurrency cap for concurrent extraction tasks.
# Each task is blocked at the Ollama semaphore anyway (Semaphore(1) in
# ollama_client.py), so this controls how many tasks queue up simultaneously
# (limits memory usage from holding many prompts in-flight).
# ---------------------------------------------------------------------------
_EXTRACTION_CONCURRENCY = 2

# Max tokens for the extraction output — always a small JSON object.
_EXTRACTION_MAX_TOKENS = 200

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a legal-text entity extractor for the South African Reserve Bank Currency and Exchanges Manual. \
You extract structured facts from ONE provision at a time. \
You do not interpret, summarise, or add information not present in the text. \
Output ONLY valid JSON matching the schema below — no preamble, no markdown fences, no commentary.

Schema:
{
  "defined_terms_used": [string],
  "cross_references": [string],
  "limits": [
    {
      "amount": number,
      "currency": "ZAR" | "<other currency code>",
      "period": "per_calendar_year" | "per_transaction" | "per_calendar_month" | "per_day" | "once_off" | "unspecified",
      "description": string
    }
  ],
  "party_roles": [string],
  "legal_instruments": [string],
  "entities": [string]
}

Rules:
- If a field has no matches, return an empty array — never omit a key, never invent a value.
- "cross_references" captures explicit internal citations only ("see section X"), not general topic similarity.
- Do not extract limits from surrounding/adjacent clauses — only this text.
- Preserve amounts and currency exactly as written (R2 million -> 2000000 ZAR, not 2 or 2000).
- "party_roles" uses ONLY: Resident, NonResident, Minor, Student, Immigrant, ProspectiveImmigrant, ForeignNational, CMAResident, AffectedPerson, Spouse, Other.
- "defined_terms_used": Capitalised terms from the manual's glossary that appear in this text.
- "entities": Named institutions mentioned by name (banks, ADLAs) — not generic terms like "Authorised Dealer".

One-shot example:
Input: "Prospective immigrants and immigrants who have applied for, but who have not been granted permanent residence in South Africa may be granted a travel allowance within the single discretionary allowance limit of R2 million per calendar year."
Output:
{
  "defined_terms_used": ["single discretionary allowance"],
  "cross_references": [],
  "limits": [
    {"amount": 2000000, "currency": "ZAR", "period": "per_calendar_year", "description": "travel allowance within the single discretionary allowance"}
  ],
  "party_roles": ["ProspectiveImmigrant", "Immigrant"],
  "legal_instruments": [],
  "entities": []
}"""

_USER_TEMPLATE = """\
Path: {path}
Heading trail: {heading_trail}
Text:
\"\"\"{text}\"\"\""""

# Valid period values for normalisation
_VALID_PERIODS = frozenset([
    "per_calendar_year", "per_transaction", "per_calendar_month",
    "per_day", "once_off", "unspecified",
])

# Valid party roles (closed set)
_VALID_PARTY_ROLES = frozenset([
    "Resident", "NonResident", "Minor", "Student", "Immigrant",
    "ProspectiveImmigrant", "ForeignNational", "CMAResident",
    "AffectedPerson", "Spouse", "Other",
])


# ---------------------------------------------------------------------------
# Extraction result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExtractionOutput:
    defined_terms_used: list[str]
    cross_references: list[str]
    limits: list[dict]
    party_roles: list[str]
    legal_instruments: list[str]
    entities: list[str]


_EMPTY_EXTRACTION = ExtractionOutput(
    defined_terms_used=[],
    cross_references=[],
    limits=[],
    party_roles=[],
    legal_instruments=[],
    entities=[],
)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_extraction_prompt(chunk: ProvisionChunk) -> str:
    """Return the full prompt (system + user) for one provision chunk."""
    heading_trail_str = " > ".join(chunk.heading_trail)
    user_part = _USER_TEMPLATE.format(
        path=chunk.path,
        heading_trail=heading_trail_str,
        text=chunk.text,
    )
    # Ollama /api/generate uses a single "prompt" field; we embed the system
    # instruction at the top, separated by a clear delimiter.
    return f"{_SYSTEM_PROMPT}\n\n---\n\n{user_part}"


# ---------------------------------------------------------------------------
# LLM call + JSON parsing
# ---------------------------------------------------------------------------

async def extract_provision(
    chunk: ProvisionChunk,
    client: OllamaClient,
) -> ExtractionOutput:
    """Call Mistral 7B to extract structured facts from one provision chunk.

    Uses json_mode=True (Ollama format:"json") to constrain decoding, and
    caps output at _EXTRACTION_MAX_TOKENS since the output is always small.

    Returns _EMPTY_EXTRACTION on any error so a single bad chunk does not
    abort the whole ingestion run.
    """
    prompt = build_extraction_prompt(chunk)
    try:
        raw = await client.generate(
            prompt,
            json_mode=True,
            max_tokens=_EXTRACTION_MAX_TOKENS,
        )
    except OllamaClientError as exc:
        logger.error(
            "Ollama error extracting provision %s: %s — skipping.",
            chunk.path, exc,
        )
        return _EMPTY_EXTRACTION

    return _parse_extraction_response(raw, chunk.path)


def _parse_extraction_response(raw: str, path: str) -> ExtractionOutput:
    """Parse and validate the LLM JSON response.

    Strips markdown fences if present.  Returns _EMPTY_EXTRACTION on parse
    failure so one bad response does not abort the ingestion run.
    """
    text = raw.strip()
    # Strip optional markdown code fences.
    if text.startswith("```"):
        lines = text.splitlines()
        end = -1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[1:end]).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error for provision %s: %s — raw: %.200s", path, exc, raw)
        return _EMPTY_EXTRACTION

    if not isinstance(data, dict):
        logger.warning("Non-dict JSON for provision %s: %s", path, type(data))
        return _EMPTY_EXTRACTION

    # Coerce and validate each field with sensible defaults.
    def _strs(key: str) -> list[str]:
        val = data.get(key, [])
        if not isinstance(val, list):
            return []
        return [str(v) for v in val if v]

    raw_limits = data.get("limits", [])
    limits: list[dict] = []
    if isinstance(raw_limits, list):
        for lim in raw_limits:
            if not isinstance(lim, dict):
                continue
            amount = lim.get("amount")
            try:
                amount = float(amount)
            except (TypeError, ValueError):
                continue  # skip malformed limit
            period = lim.get("period", "unspecified")
            if period not in _VALID_PERIODS:
                period = "unspecified"
            limits.append({
                "amount": amount,
                "currency": str(lim.get("currency", "ZAR")),
                "period": period,
                "description": str(lim.get("description", "")),
            })

    raw_roles = _strs("party_roles")
    party_roles = [r for r in raw_roles if r in _VALID_PARTY_ROLES]
    invalid_roles = [r for r in raw_roles if r not in _VALID_PARTY_ROLES]
    if invalid_roles:
        logger.debug(
            "Provision %s: ignoring invalid party roles: %s", path, invalid_roles
        )

    return ExtractionOutput(
        defined_terms_used=_strs("defined_terms_used"),
        cross_references=_strs("cross_references"),
        limits=limits,
        party_roles=party_roles,
        legal_instruments=_strs("legal_instruments"),
        entities=_strs("entities"),
    )


# ---------------------------------------------------------------------------
# Neo4j MERGE loader
# ---------------------------------------------------------------------------

class Neo4jLoader:
    """Writes parsed provision and definition chunks into Neo4j.

    All writes use MERGE on the unique key for each label, making ingestion
    fully idempotent and re-runnable (re-uploading the same manual only
    updates existing nodes, it does not duplicate them).

    Parameters
    ----------
    driver:
        An open synchronous Neo4j driver.
    manual_name:
        The ``name`` property for the Manual root node.
    manual_version:
        Version string stored on the Manual node (e.g. "2024-01").
    manual_issue_date:
        Issue date string stored on the Manual node.
    """

    def __init__(
        self,
        driver: Driver,
        manual_name: str,
        manual_version: str = "unknown",
        manual_issue_date: str = "unknown",
    ) -> None:
        self._driver = driver
        self._manual_name = manual_name
        self._manual_version = manual_version
        self._manual_issue_date = manual_issue_date

    # ------------------------------------------------------------------
    # Top-level entry points
    # ------------------------------------------------------------------

    def ensure_manual_node(self) -> None:
        """MERGE the singleton Manual root node."""
        cypher = """
            MERGE (m:Manual {name: $name})
            ON CREATE SET m.version = $version, m.issue_date = $issue_date
            ON MATCH  SET m.version = $version, m.issue_date = $issue_date
        """
        with self._driver.session() as session:
            session.run(
                cypher,
                name=self._manual_name,
                version=self._manual_version,
                issue_date=self._manual_issue_date,
            )

    def merge_provision(self, chunk: ProvisionChunk) -> None:
        """MERGE a Provision node and wire it to its parent (or to Manual)."""
        cypher = """
            MERGE (p:Provision {path: $path})
            ON CREATE SET p.level     = $level,
                          p.heading   = $heading,
                          p.text      = $text
            ON MATCH  SET p.level     = $level,
                          p.heading   = $heading,
                          p.text      = $text
        """
        heading = chunk.heading_trail[-1] if chunk.heading_trail else chunk.path
        with self._driver.session() as session:
            session.run(
                cypher,
                path=chunk.path,
                level=chunk.level,
                heading=heading,
                text=chunk.text,
            )
            # Wire to parent.
            if chunk.level == 0:
                session.run(
                    """
                    MATCH (m:Manual {name: $manual_name})
                    MATCH (p:Provision {path: $path})
                    MERGE (m)-[:HAS_TOP_SECTION]->(p)
                    """,
                    manual_name=self._manual_name,
                    path=chunk.path,
                )
            else:
                # Parent path = everything before the last marker segment.
                parent_path = _extract_parent_path(chunk.path)
                if parent_path:
                    session.run(
                        """
                        MATCH (parent:Provision {path: $parent_path})
                        MATCH (child:Provision  {path: $child_path})
                        MERGE (parent)-[:HAS_CHILD]->(child)
                        """,
                        parent_path=parent_path,
                        child_path=chunk.path,
                    )

    def merge_definition(self, defn: DefinitionChunk) -> None:
        """MERGE a Definition node."""
        cypher = """
            MERGE (d:Definition {term: $term})
            ON CREATE SET d.text = $text, d.path = $path
            ON MATCH  SET d.text = $text, d.path = $path
        """
        with self._driver.session() as session:
            session.run(cypher, term=defn.term, text=defn.text, path=defn.path)

    def apply_extraction(
        self, chunk: ProvisionChunk, extraction: ExtractionOutput
    ) -> None:
        """Write all entities/relationships extracted from one provision chunk."""
        with self._driver.session() as session:
            # USES_TERM → Definition
            for term in extraction.defined_terms_used:
                session.run(
                    """
                    MATCH (p:Provision {path: $path})
                    MERGE (d:Definition {term: $term})
                    MERGE (p)-[:USES_TERM]->(d)
                    """,
                    path=chunk.path,
                    term=term,
                )

            # CROSS_REFERENCES → Provision (MERGE the target if not yet written)
            for ref_path in extraction.cross_references:
                session.run(
                    """
                    MATCH (src:Provision {path: $src_path})
                    MERGE (tgt:Provision {path: $tgt_path})
                    MERGE (src)-[:CROSS_REFERENCES]->(tgt)
                    """,
                    src_path=chunk.path,
                    tgt_path=ref_path,
                )

            # SPECIFIES_LIMIT → Limit
            for lim in extraction.limits:
                # Limit nodes are not uniquely keyed — create one per
                # (provision, description) pair to avoid merging distinct limits.
                key = f"{chunk.path}|{lim['description'][:80]}"
                session.run(
                    """
                    MATCH (p:Provision {path: $path})
                    MERGE (l:Limit {_key: $key})
                    ON CREATE SET l.amount      = $amount,
                                  l.currency    = $currency,
                                  l.period      = $period,
                                  l.description = $description
                    ON MATCH  SET l.amount      = $amount,
                                  l.currency    = $currency,
                                  l.period      = $period,
                                  l.description = $description
                    MERGE (p)-[:SPECIFIES_LIMIT]->(l)
                    """,
                    path=chunk.path,
                    key=key,
                    amount=lim["amount"],
                    currency=lim["currency"],
                    period=lim["period"],
                    description=lim["description"],
                )

            # APPLIES_TO → PartyRole (must already exist — seeded at startup)
            for role in extraction.party_roles:
                session.run(
                    """
                    MATCH (p:Provision  {path: $path})
                    MATCH (r:PartyRole  {name: $role})
                    MERGE (p)-[:APPLIES_TO]->(r)
                    """,
                    path=chunk.path,
                    role=role,
                )

            # REFERENCES_LAW → LegalInstrument
            for instrument in extraction.legal_instruments:
                session.run(
                    """
                    MATCH (p:Provision {path: $path})
                    MERGE (l:LegalInstrument {name: $name})
                    MERGE (p)-[:REFERENCES_LAW]->(l)
                    """,
                    path=chunk.path,
                    name=instrument,
                )

            # AUTHORISED_UNDER ← Entity (institution mentioned in this provision)
            for entity_name in extraction.entities:
                session.run(
                    """
                    MATCH (p:Provision {path: $path})
                    MERGE (e:Entity {name: $name})
                    MERGE (e)-[:AUTHORISED_UNDER]->(p)
                    """,
                    path=chunk.path,
                    name=entity_name,
                )


# ---------------------------------------------------------------------------
# Concurrent extraction runner
# ---------------------------------------------------------------------------

async def extract_and_load_all(
    provisions: list[ProvisionChunk],
    definitions: list[DefinitionChunk],
    client: OllamaClient,
    loader: Neo4jLoader,
) -> None:
    """Run extraction for all provisions concurrently, then write to Neo4j.

    Phase 1 — Merge all Provision and Definition nodes (no extraction yet).
              This ensures cross-reference targets exist before we write edges.
    Phase 2 — Extract + write edges for each provision.

    Concurrency is bounded by _EXTRACTION_CONCURRENCY (tasks) and ultimately
    by the OllamaClient semaphore (one Ollama call at a time).
    """
    logger.info(
        "extract_and_load_all: loading %d provisions, %d definitions.",
        len(provisions), len(definitions),
    )

    # Phase 1: write all structural nodes first (fast — no LLM calls).
    loader.ensure_manual_node()
    for chunk in provisions:
        try:
            loader.merge_provision(chunk)
        except Neo4jError as exc:
            logger.error("Neo4j error merging provision %s: %s", chunk.path, exc)

    for defn in definitions:
        try:
            loader.merge_definition(defn)
        except Neo4jError as exc:
            logger.error("Neo4j error merging definition %s: %s", defn.term, exc)

    logger.info("Phase 1 complete: structural nodes written.")

    # Phase 2: concurrent extraction + edge writes.
    semaphore = asyncio.Semaphore(_EXTRACTION_CONCURRENCY)

    async def _extract_one(chunk: ProvisionChunk) -> None:
        async with semaphore:
            extraction = await extract_provision(chunk, client)
        try:
            loader.apply_extraction(chunk, extraction)
        except Neo4jError as exc:
            logger.error(
                "Neo4j error writing extraction for %s: %s", chunk.path, exc
            )

    await asyncio.gather(*[_extract_one(c) for c in provisions])
    logger.info("Phase 2 complete: extraction edges written.")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# Matches the last segment of a compound path, e.g. "(bb)" in "B.4(A)(ii)(a)(bb)"
_RE_LAST_SEGMENT = re.compile(r"\([^)]+\)$")


def _extract_parent_path(path: str) -> str | None:
    """Return the path of the parent provision, or None for top-level paths.

    Examples:
      "B.4(A)(ii)(a)(bb)" → "B.4(A)(ii)(a)"
      "B.4(A)"            → "B.4"
      "B.4"               → None (top-level, parent is Manual)
    """
    # Top-level path like "B.4" has no parenthesised segment.
    m = _RE_LAST_SEGMENT.search(path)
    if m is None:
        return None
    return path[: m.start()]
