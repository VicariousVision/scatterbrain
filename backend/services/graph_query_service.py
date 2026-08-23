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
from typing import Any, Optional

from neo4j import GraphDatabase, Driver
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.retrievers import Text2CypherRetriever

logger = logging.getLogger(__name__)

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

    When a paid-tier API key is configured (ANTHROPIC_API_KEY, DEEPSEEK_API_KEY,
    or OPENAI_API_KEY) that model is preferred.  Otherwise the provided
    ``fallback_llm`` (an OllamaLLMAdapter wrapping the local Ollama model) is
    used so that Text2CypherRetriever is always available.

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

    async def get_relevant_context(self, query: str, top_k: int = 5) -> str:
        """Return formatted Cypher-query results for a user question.

        Parameters
        ----------
        query:   The user's natural-language question.
        top_k:   Passed to the retriever as the result limit.

        Returns
        -------
        str
            Formatted context string, or a fallback message if nothing found.
        """
        if self._retriever is None:
            return (
                "Knowledge graph retrieval is unavailable — no LLM is configured. "
                "Check that Ollama is running or set a paid-tier API key."
            )

        import asyncio
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self._retriever.search(query_text=query),
            )
        except Exception as exc:
            exc_msg = str(exc)
            logger.error("Text2CypherRetriever.search failed: %s", exc_msg)

            # Surface a more helpful message when the LLM clearly produced
            # garbage instead of valid Cypher.
            if "Unexpected end of input" in exc_msg or "expected CYPHER" in exc_msg:
                logger.error(
                    "The Text2Cypher LLM likely returned invalid output instead "
                    "of a Cypher query.  Consider using a larger model for "
                    "ollama_text2cypher_model (e.g. qwen3:1.7b or mistral)."
                )
            return "No relevant graph data found for this query."

        if not result.items:
            return "No relevant graph data found for this query."

        return "\n\n---\n\n".join(item.content for item in result.items)

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        self._driver.close()

    def __enter__(self) -> "GraphQueryService":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
