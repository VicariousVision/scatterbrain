"""Graph query service using neo4j-graphrag Text2CypherRetriever.

Converts natural-language questions directly to Cypher via an LLM — no
embeddings, no vector index.  The paid-tier LLM (Claude Haiku or DeepSeek
via OpenAI-compatible API) is used for Cypher generation when available.
If no paid-tier key is configured, falls back to the local Ollama model
specified by ``ollama_text2cypher_model`` in settings (default: qwen3.5:0.8b).

The retriever is configured with:
  - The Neo4j driver
  - An LLMInterface pointing at the selected model
  - A schema description derived from the domain graph labels/relationships
  - Hand-written NL→Cypher example pairs covering every relationship type

No VectorRetriever, VectorCypherRetriever, embedder, or vector index is
referenced anywhere in this file.

Requirements: query-time retrieval (Step 5)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

from neo4j import GraphDatabase, Driver
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.retrievers import Text2CypherRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cypher post-processing helpers
# ---------------------------------------------------------------------------

def _sanitize_cypher(cypher: str) -> str:
    """Clean up common LLM Cypher generation errors.

    Problems seen from free-tier models:
      1. ``ORDER BY p`` — ordering by a node variable instead of a property;
         invalid after WITH DISTINCT or aggregation because ``p`` is dropped.
      2. Markdown fences (```cypher ... ```) wrapped around the query.
      3. Trailing prose after the last semicolon or newline.
      4. ``None`` returned by the model (null content) — treated as empty.

    Strategy:
      - Return empty string for None/empty input.
      - Strip markdown fences.
      - Remove ``ORDER BY`` clauses that reference bare node/rel variables
        (single identifiers with no dot-property accessor).  Property
        accesses like ``ORDER BY p.level, p.path`` are kept intact.
    """
    if not cypher:
        return ""
    # 1. Strip markdown code fences.
    cypher = re.sub(r"```[a-zA-Z]*\n?", "", cypher).strip("`").strip()

    # 2. Remove ORDER BY entries that are bare identifiers (no dot access).
    #    e.g. "ORDER BY p.level, p" → "ORDER BY p.level"
    #    e.g. "ORDER BY p"          → removed entirely
    def _clean_order_by(match: re.Match) -> str:
        clause = match.group(0)
        # Split on comma, keep only items that contain a dot (property access)
        # or are numeric / string literals.
        items = [item.strip() for item in re.split(r",", clause[len("ORDER BY"):], flags=re.IGNORECASE)]
        kept = [
            item for item in items
            if "." in item  # property access: p.level, p.path, etc.
            or re.match(r"^['\"]", item)  # string literal
            or re.match(r"^\d", item)     # numeric literal
        ]
        if not kept:
            return ""
        return "ORDER BY " + ", ".join(kept)

    cypher = re.sub(
        r"\bORDER\s+BY\s+[^\n]+",
        _clean_order_by,
        cypher,
        flags=re.IGNORECASE,
    )

    # 3. Collapse multiple blank lines left by removal.
    cypher = re.sub(r"\n{3,}", "\n\n", cypher).strip()

    return cypher


# ---------------------------------------------------------------------------
# Fallback: direct keyword search when Text2Cypher fails
# ---------------------------------------------------------------------------

_KEYWORD_FALLBACK_QUERY = """
MATCH (p:Provision)
WHERE {where_clause}
RETURN p.path AS path, p.heading AS heading, p.text AS text
ORDER BY p.level, p.path
LIMIT {limit}
"""

_CAPITAL_KEYWORDS = [
    "capital", "foreign", "allowance", "limit", "transfer",
    "invest", "abroad", "offshore", "remit",
]

def _extract_keywords(query: str, min_length: int = 4) -> list[str]:
    """Extract meaningful words from *query* as search terms."""
    stop = {
        "what", "are", "the", "for", "and", "with", "that", "this",
        "from", "have", "has", "can", "does", "will", "how", "which",
        "when", "where", "is", "in", "of", "to", "a", "an", "do",
        "company", "moving", "outside", "south", "africa",
    }
    words = re.findall(r"[a-zA-Z]+", query.lower())
    return [w for w in words if len(w) >= min_length and w not in stop]


def _build_fallback_cypher(query: str, limit: int = 5) -> str | None:
    """Build a simple CONTAINS-based Cypher query from *query* keywords.

    Returns None if no usable keywords were found.
    """
    keywords = _extract_keywords(query)
    if not keywords:
        return None

    # Use at most the 4 most relevant keywords to avoid over-filtering.
    terms = keywords[:4]
    conditions = " OR ".join(
        f"toLower(p.text) CONTAINS '{term}' OR toLower(p.heading) CONTAINS '{term}'"
        for term in terms
    )
    return _KEYWORD_FALLBACK_QUERY.format(where_clause=conditions, limit=limit).strip()

# ---------------------------------------------------------------------------
# Schema description passed to Text2CypherRetriever
# ---------------------------------------------------------------------------
# This is a human-readable description of the graph schema.  The retriever
# includes it in the Cypher-generation prompt so the LLM knows what labels
# and relationships exist.

_NEO4J_SCHEMA = """
Node labels and key properties:
  Manual         {name, version, issue_date}
  Provision      {path (unique), level, heading, text}
  Definition     {term (unique), text, path}
  Limit          {amount, currency, period, description}
  PartyRole      {name}  -- closed set: Resident, NonResident, Minor, Student,
                            Immigrant, ProspectiveImmigrant, ForeignNational,
                            CMAResident, AffectedPerson, Spouse, Other
  Entity         {name (unique), type, category}  -- type: AuthorisedDealer |
                            RestrictedAuthorisedDealer | ADLA
  LegalInstrument {name (unique), act_no, year}

Relationships:
  (:Manual)-[:HAS_TOP_SECTION]->(:Provision {level: 0})
  (:Provision)-[:HAS_CHILD]->(:Provision)
  (:Provision)-[:CROSS_REFERENCES]->(:Provision)
  (:Provision)-[:USES_TERM]->(:Definition)
  (:Provision)-[:SPECIFIES_LIMIT]->(:Limit)
  (:Provision)-[:APPLIES_TO]->(:PartyRole)
  (:Provision)-[:REFERENCES_LAW]->(:LegalInstrument)
  (:Entity)-[:AUTHORISED_UNDER]->(:Provision)
  (:Definition)-[:REFERENCES_TERM]->(:Definition)

Important notes:
  - p.path is the citation key (e.g. "B.4(B)(iv)(d)(bb)") — always RETURN it.
  - p.level 0 = top section, 1 = subsection, 2 = clause, 3 = sub-clause, etc.
  - Use CONTAINS for partial text searches on p.text or p.heading.
  - Always RETURN p.path and p.text so the answer is citable.
"""

# ---------------------------------------------------------------------------
# Hand-written NL → Cypher examples
# One per relationship type to anchor Text2Cypher accuracy.
# ---------------------------------------------------------------------------

_EXAMPLES_RAW: list[dict[str, str]] = [
    # SPECIFIES_LIMIT
    {
        "question": "What is the single discretionary allowance limit for a minor travelling abroad?",
        "cypher": (
            "MATCH (p:Provision)-[:APPLIES_TO]->(:PartyRole {name: 'Minor'}), "
            "      (p)-[:SPECIFIES_LIMIT]->(l:Limit) "
            "WHERE p.text CONTAINS 'travel allowance' "
            "RETURN p.path, p.text, l.amount, l.currency, l.period, l.description"
        ),
    },
    # APPLIES_TO
    {
        "question": "Which provisions apply to non-residents?",
        "cypher": (
            "MATCH (p:Provision)-[:APPLIES_TO]->(:PartyRole {name: 'NonResident'}) "
            "RETURN p.path, p.heading, p.text ORDER BY p.level, p.path"
        ),
    },
    # CROSS_REFERENCES
    {
        "question": "Which provisions reference section B.5(A)?",
        "cypher": (
            "MATCH (src:Provision)-[:CROSS_REFERENCES]->(tgt:Provision {path: 'B.5(A)'}) "
            "RETURN src.path, src.heading, src.text"
        ),
    },
    # USES_TERM
    {
        "question": "What does 'CMA' mean and which provisions use it?",
        "cypher": (
            "MATCH (d:Definition) WHERE d.term CONTAINS 'CMA' "
            "OPTIONAL MATCH (p:Provision)-[:USES_TERM]->(d) "
            "RETURN d.term, d.text, collect(p.path) AS used_in_provisions"
        ),
    },
    # AUTHORISED_UNDER
    {
        "question": "Which provisions authorise ABSA Bank?",
        "cypher": (
            "MATCH (e:Entity {name: 'ABSA Bank'})-[:AUTHORISED_UNDER]->(p:Provision) "
            "RETURN p.path, p.heading, p.text"
        ),
    },
    # REFERENCES_LAW
    {
        "question": "Which provisions reference the Financial Intelligence Centre Act?",
        "cypher": (
            "MATCH (p:Provision)-[:REFERENCES_LAW]->(l:LegalInstrument) "
            "WHERE l.name CONTAINS 'Financial Intelligence Centre' "
            "RETURN p.path, p.text, l.name"
        ),
    },
    # HAS_CHILD (hierarchy traversal)
    {
        "question": "List all subsections of section B.4.",
        "cypher": (
            "MATCH (parent:Provision {path: 'B.4'})-[:HAS_CHILD]->(child:Provision) "
            "RETURN child.path, child.heading ORDER BY child.path"
        ),
    },
    # Limit with currency and period
    {
        "question": "What are all the ZAR annual limits for residents?",
        "cypher": (
            "MATCH (p:Provision)-[:APPLIES_TO]->(:PartyRole {name: 'Resident'}), "
            "      (p)-[:SPECIFIES_LIMIT]->(l:Limit {currency: 'ZAR', period: 'per_calendar_year'}) "
            "RETURN p.path, p.text, l.amount, l.description ORDER BY l.amount DESC"
        ),
    },
    # Full-text search across the whole manual
    {
        "question": "Find all provisions that mention 'foreign capital allowance'.",
        "cypher": (
            "MATCH (p:Provision) "
            "WHERE toLower(p.text) CONTAINS 'foreign capital allowance' "
            "RETURN p.path, p.heading, p.text ORDER BY p.level, p.path"
        ),
    },
    # Definition lookup
    {
        "question": "What is the definition of 'Authorised Dealer'?",
        "cypher": (
            "MATCH (d:Definition) "
            "WHERE toLower(d.term) CONTAINS 'authorised dealer' "
            "RETURN d.term, d.text"
        ),
    },
]

# Text2CypherRetriever expects examples as plain strings, not dicts.
_EXAMPLES: list[str] = [
    f"Question: {ex['question']}\nCypher: {ex['cypher']}"
    for ex in _EXAMPLES_RAW
]


# ---------------------------------------------------------------------------
# Paid-tier LLM wrapper for Cypher generation
# ---------------------------------------------------------------------------

def _build_paid_llm(fallback_llm: Optional[LLMInterface] = None) -> LLMInterface:
    """Build an LLMInterface for Cypher generation.

    Checks env vars to decide which provider to use, in priority order:
      1. ANTHROPIC_API_KEY  → Claude claude-haiku-4-5 via neo4j_graphrag AnthropicLLM
      2. DEEPSEEK_API_KEY   → DeepSeek Chat via neo4j_graphrag OpenAILLM
      3. OPENAI_API_KEY     → OpenAI-compatible model via neo4j_graphrag OpenAILLM
      4. fallback_llm       → Local Ollama model (OllamaLLMAdapter)

    Parameters
    ----------
    fallback_llm:
        An LLMInterface (typically OllamaLLMAdapter) to use when no paid-tier
        API key is configured.  Must not be None — the caller is responsible
        for always providing a local fallback.

    Returns
    -------
    LLMInterface
        The selected LLM, guaranteed to be non-None.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")

    if anthropic_key:
        try:
            from neo4j_graphrag.llm import AnthropicLLM  # type: ignore[import]
            logger.info("Text2Cypher: using Claude claude-haiku-4-5 (Anthropic).")
            return AnthropicLLM(
                model_name="claude-haiku-4-5",
                api_key=anthropic_key,
            )
        except ImportError:
            logger.warning(
                "neo4j_graphrag.llm.AnthropicLLM not available. "
                "Install neo4j-graphrag[anthropic] to use Claude. "
                "Falling back to next option."
            )

    if deepseek_key or openai_key:
        key = deepseek_key or openai_key
        model = "deepseek-chat"  # DeepSeek V3 / Chat endpoint
        base_url = "https://api.deepseek.com" if deepseek_key else None
        try:
            from neo4j_graphrag.llm import OpenAILLM  # type: ignore[import]
            kwargs: dict[str, Any] = {"model_name": model, "api_key": key}
            if base_url:
                kwargs["base_url"] = base_url
            logger.info("Text2Cypher: using DeepSeek (%s).", model)
            return OpenAILLM(**kwargs)
        except ImportError:
            logger.warning(
                "neo4j_graphrag.llm.OpenAILLM not available. "
                "Install neo4j-graphrag[openai] to use DeepSeek / OpenAI. "
                "Falling back to local Ollama model."
            )

    # No paid-tier key is set (or imports failed) — use the local Ollama model.
    logger.info(
        "No paid-tier LLM API key found (ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, "
        "OPENAI_API_KEY).  Text2Cypher will use the local Ollama fallback model."
    )
    return fallback_llm


# ---------------------------------------------------------------------------
# GraphQueryService
# ---------------------------------------------------------------------------


class GraphQueryService:
    """Retrieves relevant context for chat queries using Text2CypherRetriever.

    Converts natural-language questions to Cypher via an LLM and executes the
    generated query against Neo4j.  No embedder or vector index is used.

    The default (Ollama) retriever is built at construction time.
    Per-request external LLM retrievers (DeepSeek / OpenRouter) are built
    on-demand and cached for the lifetime of this service.

    Parameters
    ----------
    uri:          Neo4j Bolt/Neo4j URI.
    username:     Neo4j username.
    password:     Neo4j password.
    fallback_llm: LLMInterface to use when no paid-tier key is available.
                  Typically an OllamaLLMAdapter pointing at qwen3.5:0.8b.
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        fallback_llm: Optional[LLMInterface] = None,
    ) -> None:
        self._driver: Driver = GraphDatabase.driver(uri, auth=(username, password))
        self._uri = uri
        self._username = username
        self._password = password

        # Default (auto-select) retriever — built at startup.
        llm = _build_paid_llm(fallback_llm=fallback_llm)
        if llm is not None:
            self._retriever: Text2CypherRetriever | None = Text2CypherRetriever(
                driver=self._driver,
                llm=llm,
                neo4j_schema=_NEO4J_SCHEMA,
                examples=_EXAMPLES,
            )
        else:
            self._retriever = None
            logger.warning(
                "GraphQueryService: Text2CypherRetriever unavailable (no LLM configured). "
                "Chat queries will return an error message."
            )

        # Cache for per-backend retrievers built on demand.
        self._external_retrievers: dict[str, Text2CypherRetriever] = {}

    def _get_retriever_for_backend(
        self, backend: str
    ) -> "Text2CypherRetriever | None":
        """Return (or build) the Text2CypherRetriever for *backend*.

        ``backend`` is one of: ``"ollama"`` (default), ``"deepseek"``,
        ``"openrouter"``.  Returns the default retriever for ``"ollama"`` or
        any unknown value.
        """
        if backend in ("ollama", ""):
            return self._retriever

        if backend in self._external_retrievers:
            return self._external_retrievers[backend]

        # Build on first use.
        try:
            if backend == "deepseek":
                from backend.config import settings as _settings
                from backend.services.external_llm_client import DeepSeekClient
                from backend.services.external_llm_adapters import DeepSeekLLMAdapter

                if not _settings.deepseek_chat_api_key:
                    logger.warning(
                        "DeepSeek backend selected but DEEPSEEK_CHAT_API_KEY is not set. "
                        "Falling back to default retriever."
                    )
                    return self._retriever
                ds_client = DeepSeekClient(
                    api_key=_settings.deepseek_chat_api_key,
                    model=_settings.deepseek_chat_model,
                )
                ds_adapter = DeepSeekLLMAdapter(
                    client=ds_client,
                    model_name=_settings.deepseek_chat_model,
                )
                retriever = Text2CypherRetriever(
                    driver=self._driver,
                    llm=ds_adapter,
                    neo4j_schema=_NEO4J_SCHEMA,
                    examples=_EXAMPLES,
                )

            elif backend == "openrouter":
                from backend.config import settings as _settings
                from backend.services.external_llm_client import OpenRouterClient
                from backend.services.external_llm_adapters import OpenRouterLLMAdapter

                if not _settings.openrouter_api_key:
                    logger.warning(
                        "OpenRouter backend selected but OPENROUTER_API_KEY is not set. "
                        "Falling back to default retriever."
                    )
                    return self._retriever
                or_client = OpenRouterClient(
                    api_key=_settings.openrouter_api_key,
                    site_url=_settings.openrouter_site_url,
                    site_name=_settings.openrouter_site_name,
                )
                or_adapter = OpenRouterLLMAdapter(
                    client=or_client,
                    model_name="openrouter",
                )
                retriever = Text2CypherRetriever(
                    driver=self._driver,
                    llm=or_adapter,
                    neo4j_schema=_NEO4J_SCHEMA,
                    examples=_EXAMPLES,
                )
            else:
                return self._retriever

            self._external_retrievers[backend] = retriever
            logger.info("GraphQueryService: built Text2CypherRetriever for backend=%s.", backend)
            return retriever

        except Exception as exc:
            logger.error(
                "Failed to build %s Text2CypherRetriever: %s. Falling back to default.",
                backend,
                exc,
            )
            return self._retriever

    async def get_relevant_context(
        self, query: str, top_k: int = 5, backend: str = "ollama"
    ) -> tuple[str, str | None, str | None]:
        """Return formatted Cypher-query results for a user question.

        Parameters
        ----------
        query:   The user's natural-language question.
        top_k:   Passed to the retriever as the result limit.
        backend: One of ``"ollama"``, ``"deepseek"``, ``"openrouter"``.
                 Controls which LLM is used for Cypher generation.

        Returns
        -------
        tuple[str, str | None, str | None]
            ``(context, generated_cypher, cypher_source)`` where:
            - ``context`` is the formatted graph results string
            - ``generated_cypher`` is the Cypher that was executed (or None)
            - ``cypher_source`` is ``"text2cypher"``, ``"keyword_fallback"``,
              or None when retrieval was unavailable
        """
        retriever = self._get_retriever_for_backend(backend)

        if retriever is None:
            return (
                "Knowledge graph retrieval is unavailable — no LLM is configured. "
                "Check that Ollama is running or set a paid-tier API key.",
                None,
                None,
            )

        import asyncio
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: retriever.search(query_text=query),
            )
        except Exception as exc:
            exc_msg = str(exc)
            logger.error("Text2CypherRetriever.search failed (backend=%s): %s", backend, exc_msg)

            # Surface a more helpful message when the LLM clearly produced
            # garbage instead of valid Cypher.
            if "Unexpected end of input" in exc_msg or "expected CYPHER" in exc_msg:
                logger.error(
                    "The Text2Cypher LLM likely returned invalid output instead "
                    "of a Cypher query.  Consider using a larger model for "
                    "ollama_text2cypher_model (e.g. qwen3:1.7b or mistral)."
                )

            # Attempt keyword-based fallback before giving up entirely.
            logger.info(
                "Text2Cypher failed — attempting keyword fallback for query: %s", query
            )
            fallback_context, fallback_cypher = await self._keyword_fallback(query, top_k=top_k)
            if fallback_context:
                logger.info("Keyword fallback succeeded.")
                return fallback_context, fallback_cypher, "keyword_fallback"

            return "No relevant graph data found for this query.", None, None

        if not result.items:
            # Text2Cypher ran but returned nothing — try keyword fallback.
            logger.info(
                "Text2CypherRetriever returned no items — attempting keyword fallback."
            )
            fallback_context, fallback_cypher = await self._keyword_fallback(query, top_k=top_k)
            if fallback_context:
                logger.info("Keyword fallback succeeded.")
                return fallback_context, fallback_cypher, "keyword_fallback"
            return "No relevant graph data found for this query.", None, None

        # Extract the Cypher that was actually executed.  neo4j-graphrag stores
        # it on the retriever's last result metadata when available.
        executed_cypher: str | None = None
        try:
            executed_cypher = result.metadata.get("cypher") if result.metadata else None
        except Exception:
            pass

        context = "\n\n---\n\n".join(item.content for item in result.items)
        return context, executed_cypher, "text2cypher"

    async def _keyword_fallback(self, query: str, top_k: int = 5) -> tuple[str, str | None]:
        """Run a direct Neo4j CONTAINS search when Text2Cypher fails or returns nothing.

        Extracts keywords from *query* and searches ``p.text`` / ``p.heading``
        on all ``Provision`` nodes.  Returns ``(formatted_results, cypher)``
        where ``cypher`` is the query that was run, or ``("", None)`` if
        nothing was found or no keywords could be extracted.
        """
        cypher = _build_fallback_cypher(query, limit=top_k)
        if cypher is None:
            logger.warning("Keyword fallback: no usable keywords in query '%s'.", query)
            return "", None

        logger.debug("Keyword fallback Cypher:\n%s", cypher)
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            records = await loop.run_in_executor(
                None,
                lambda: self._driver.execute_query(cypher).records,
            )
        except Exception as exc:
            logger.error("Keyword fallback query failed: %s", exc)
            return "", cypher

        if not records:
            return "", cypher

        parts: list[str] = []
        for rec in records:
            path = rec.get("path", "")
            heading = rec.get("heading", "")
            text = rec.get("text", "")
            parts.append(f"[{path}] {heading}\n{text}".strip())

        return "\n\n---\n\n".join(parts), cypher

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        self._driver.close()

    def __enter__(self) -> "GraphQueryService":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
