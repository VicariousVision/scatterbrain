"""Chat service using neo4j-graphrag GraphRAG pipeline.

Replaces the hand-rolled prompt builder with the neo4j-graphrag ``GraphRAG``
class, which handles retrieval + prompt construction + LLM generation.

Requirements: 5.3, 5.4, 5.5, 6.1, 6.2, 6.4, 6.5
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from neo4j_graphrag.generation import GraphRAG, RagTemplate

from backend.services.graph_query_service import GraphQueryService
from backend.services.ollama_adapters import OllamaLLMAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
# RagTemplate variables: {query_text}, {context}, {message_history}

_LEGAL_PROMPT = """\
You are a legal document analysis assistant. \
Answer the question using ONLY the information in the provided context. \
If the context does not contain enough information to answer, say so clearly. \
Do not fabricate facts.

CONTEXT:
{context}

CONVERSATION HISTORY:
{message_history}

QUESTION:
{query_text}

ANSWER:"""

_PROMPT_TEMPLATE = RagTemplate(
    template=_LEGAL_PROMPT,
    expected_inputs=["context", "query_text", "message_history"],
)


class ChatService:
    """Orchestrates the GraphRAG pipeline for chat queries.

    Parameters
    ----------
    graph_query_service:
        Used to retrieve relevant context from the knowledge graph.
    llm_adapter:
        ``OllamaLLMAdapter`` wrapping the local Ollama model.
    """

    def __init__(
        self,
        graph_query_service: GraphQueryService,
        llm_adapter: OllamaLLMAdapter,
    ) -> None:
        self._graph_query_service = graph_query_service
        self._rag = GraphRAG(
            retriever=graph_query_service._retriever,
            llm=llm_adapter,
            prompt_template=_PROMPT_TEMPLATE,
        )

    async def query(
        self, user_query: str, history: List[Dict[str, str]]
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Process a chat query using the neo4j-graphrag GraphRAG pipeline.

        Requirements: 5.3, 5.4, 5.5, 6.1, 6.2, 6.4, 6.5

        Parameters
        ----------
        user_query:
            The user's natural language question.
        history:
            Current conversation history as list of ``{"role", "content"}`` dicts.

        Returns
        -------
        Tuple[str, List[Dict[str, str]]]
            ``(response_text, updated_history)``
        """
        # Truncate history to last 10 messages (Requirement 6.5)
        truncated = history[-10:] if len(history) > 10 else history
        message_history_str = self._format_history(truncated)

        logger.info("Running GraphRAG query: %s", user_query)

        # GraphRAG.search is synchronous; run in executor to avoid blocking.
        import asyncio
        loop = asyncio.get_event_loop()
        rag_response = await loop.run_in_executor(
            None,
            lambda: self._rag.search(
                query_text=user_query,
                retriever_config={"top_k": 5},
                prompt_params={"message_history": message_history_str},
                response_fallback=(
                    "I don't have enough information in the knowledge graph "
                    "to answer that question."
                ),
            ),
        )

        response_text = rag_response.answer

        updated_history = history + [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": response_text},
        ]
        return response_text, updated_history

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "(No previous messages)"
        return "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in history
        )
