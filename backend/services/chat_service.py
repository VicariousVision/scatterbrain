"""Chat service using Text2CypherRetriever for context retrieval.

Retrieval is handled by GraphQueryService (Text2CypherRetriever via the
paid-tier LLM).  Final answer generation uses Mistral 7B via OllamaLLMAdapter.

No embedder, no VectorRetriever, no vector index anywhere in this path.

Requirements: 5.3, 5.4, 5.5, 6.1, 6.2, 6.4, 6.5
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from backend.services.graph_query_service import GraphQueryService
from backend.services.ollama_adapters import OllamaLLMAdapter
from backend.services.ollama_client import OllamaClientError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_LEGAL_PROMPT = """\
You are a legal document analysis assistant for the South African Reserve Bank \
Currency and Exchanges Manual for Authorised Dealers.
Answer the question using ONLY the information in the provided context.
If the context does not contain enough information to answer, say so clearly. \
Do not fabricate facts. Where possible, cite the provision path (e.g. B.4(B)(iv)).

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""


class ChatService:
    """Orchestrates context retrieval and LLM answer generation.

    Parameters
    ----------
    graph_query_service:
        Retrieves relevant Cypher query results for a user question.
    llm_adapter:
        OllamaLLMAdapter wrapping Mistral 7B for final answer generation.
    """

    def __init__(
        self,
        graph_query_service: GraphQueryService,
        llm_adapter: OllamaLLMAdapter,
    ) -> None:
        self._graph_query_service = graph_query_service
        self._llm = llm_adapter

    async def query(
        self, user_query: str, history: List[Dict[str, str]]
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Process a chat query end-to-end.

        1. Retrieve relevant graph context via Text2CypherRetriever.
        2. Build a grounded prompt and generate an answer via Mistral 7B.

        Requirements: 5.3, 5.4, 5.5, 6.1, 6.2, 6.4, 6.5

        Parameters
        ----------
        user_query:  The user's natural language question.
        history:     Current conversation history.

        Returns
        -------
        (response_text, updated_history)
        """
        # Truncate history to last 10 messages (Requirement 6.5).
        truncated = history[-10:] if len(history) > 10 else history

        logger.info("ChatService.query: %s", user_query)

        # Step 1: retrieve context from the graph.
        context = await self._graph_query_service.get_relevant_context(
            user_query, top_k=5
        )

        # Step 2: build prompt and generate answer.
        prompt = _LEGAL_PROMPT.format(context=context, query=user_query)
        try:
            response_text = await self._llm._client.generate(prompt)
        except OllamaClientError as exc:
            logger.error("Ollama generation failed: %s", exc)
            raise

        updated_history = truncated + [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": response_text},
        ]
        return response_text, updated_history
