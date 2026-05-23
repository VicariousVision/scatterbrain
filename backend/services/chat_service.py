"""Chat service for orchestrating graph-RAG chat queries.

Implements the graph-RAG pipeline: retrieves relevant graph context,
constructs a grounded prompt with conversation history, and generates
LLM responses via Ollama.

Requirements: 5.3, 5.4, 5.5, 6.1, 6.2, 6.4, 6.5
"""

from __future__ import annotations

import logging
from typing import List, Dict, Tuple

from backend.services.graph_query_service import GraphQueryService
from backend.services.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class ChatService:
    """Orchestrates the graph-RAG pipeline for chat queries.

    Parameters
    ----------
    graph_query_service:
        Service for retrieving relevant graph context.
    ollama_client:
        Client for generating LLM responses.
    """

    def __init__(
        self,
        graph_query_service: GraphQueryService,
        ollama_client: OllamaClient,
    ) -> None:
        self.graph_query_service = graph_query_service
        self.ollama_client = ollama_client

    async def query(
        self, user_query: str, history: List[Dict[str, str]]
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Process a chat query using the graph-RAG pattern.

        Retrieves relevant graph context, truncates message history to the
        last 10 messages, constructs a grounded prompt, and generates an
        LLM response.

        Requirements: 5.3, 5.4, 5.5, 6.1, 6.2, 6.4, 6.5

        Parameters
        ----------
        user_query:
            The user's natural language query.
        history:
            Current chat session message history as a list of dicts with
            ``role`` (``"user"`` or ``"assistant"``) and ``content`` keys.

        Returns
        -------
        Tuple[str, List[Dict[str, str]]]
            A tuple of (response_text, updated_history) where updated_history
            includes the user query and assistant response appended.
        """
        # Step 1: Retrieve graph context
        logger.info("Retrieving graph context for query: %s", user_query)
        graph_context = await self.graph_query_service.get_relevant_context(user_query)

        # Step 2: Truncate history to last 10 messages (Requirement 6.5)
        truncated_history = history[-10:] if len(history) > 10 else history

        # Step 3: Build chat prompt (Requirements 5.4, 6.1)
        prompt = self._build_prompt(user_query, graph_context, truncated_history)

        # Step 4: Call LLM to generate response (Requirement 5.5)
        logger.info("Generating LLM response")
        response = await self.ollama_client.generate(prompt)

        # Step 5: Update history with user query and assistant response
        updated_history = history + [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": response},
        ]

        return response, updated_history

    def _build_prompt(
        self, user_query: str, graph_context: str, history: List[Dict[str, str]]
    ) -> str:
        """Build the chat prompt with system instruction, graph context, history, and query.

        Requirements: 5.4, 6.1

        Parameters
        ----------
        user_query:
            The user's natural language query.
        graph_context:
            Formatted graph triples or "no data found" message.
        history:
            Truncated message history (at most 10 messages).

        Returns
        -------
        str
            The complete prompt to send to the LLM.
        """
        # System grounding instruction (Requirement 6.1)
        system_instruction = (
            "SYSTEM: You are a legal document analysis assistant. "
            "You MUST base your answer ONLY on the graph context provided below. "
            "Do not fabricate information that is not present in the context. "
            "If the context does not contain relevant information, state that clearly."
        )

        # Format conversation history
        history_text = self._format_history(history)

        # Construct full prompt
        prompt = f"""{system_instruction}

GRAPH CONTEXT:
{graph_context}

CONVERSATION HISTORY:
{history_text}

USER QUERY:
{user_query}

ASSISTANT:"""

        return prompt

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Format message history for inclusion in the prompt.

        Parameters
        ----------
        history:
            List of message dicts with ``role`` and ``content`` keys.

        Returns
        -------
        str
            Formatted history string, or "(No previous messages)" if empty.
        """
        if not history:
            return "(No previous messages)"

        formatted_messages = []
        for msg in history:
            role = msg["role"].upper()
            content = msg["content"]
            formatted_messages.append(f"{role}: {content}")

        return "\n".join(formatted_messages)
