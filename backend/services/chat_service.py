"""Chat service using Text2CypherRetriever for context retrieval.

Retrieval is handled by GraphQueryService (Text2CypherRetriever).  Final
answer generation is dispatched to one of three backends based on the
``backend`` parameter supplied by the caller:

  - ``"ollama"``      — local Mistral 7B via OllamaLLMAdapter (default)
  - ``"deepseek"``    — DeepSeek Chat API (DEEPSEEK_CHAT_API_KEY required)
  - ``"openrouter"``  — OpenRouter API (OPENROUTER_API_KEY required),
                        free-tier models prioritised

Both the Cypher-generation step (Text2CypherRetriever) and the answer-
synthesis step respect the selected backend.

No embedder, no VectorRetriever, no vector index anywhere in this path.

Requirements: 5.3, 5.4, 5.5, 6.1, 6.2, 6.4, 6.5
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from backend.services.graph_query_service import GraphQueryService
from backend.services.ollama_adapters import OllamaLLMAdapter
from backend.services.ollama_client import OllamaClientError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template (used by Ollama path; external backends build their own)
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
        OllamaLLMAdapter wrapping Mistral 7B for default (Ollama) answer generation.
    """

    def __init__(
        self,
        graph_query_service: GraphQueryService,
        llm_adapter: OllamaLLMAdapter,
    ) -> None:
        self._graph_query_service = graph_query_service
        self._llm = llm_adapter

        # Lazily-initialised external clients (built on first use per backend).
        self._deepseek_client = None
        self._openrouter_client = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_deepseek_client(self):
        if self._deepseek_client is None:
            from backend.config import settings
            from backend.services.external_llm_client import DeepSeekClient

            if not settings.deepseek_chat_api_key:
                raise RuntimeError(
                    "DeepSeek backend selected but DEEPSEEK_CHAT_API_KEY is not set in .env."
                )
            self._deepseek_client = DeepSeekClient(
                api_key=settings.deepseek_chat_api_key,
                model=settings.deepseek_chat_model,
            )
        return self._deepseek_client

    def _get_openrouter_client(self):
        if self._openrouter_client is None:
            from backend.config import settings
            from backend.services.external_llm_client import OpenRouterClient

            if not settings.openrouter_api_key:
                raise RuntimeError(
                    "OpenRouter backend selected but OPENROUTER_API_KEY is not set in .env."
                )
            self._openrouter_client = OpenRouterClient(
                api_key=settings.openrouter_api_key,
                site_url=settings.openrouter_site_url,
                site_name=settings.openrouter_site_name,
            )
        return self._openrouter_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def query(
        self,
        user_query: str,
        history: List[Dict[str, str]],
        backend: str = "ollama",
        rag_mode: str = "graphrag",
    ) -> Tuple[str, List[Dict[str, str]], Optional[str], Optional[str]]:
        """Process a chat query end-to-end.

        1. Retrieve relevant graph context via Text2CypherRetriever (using
           the LLM selected by *backend*).
        2. Build a grounded prompt and generate an answer using *backend*.

        Requirements: 5.3, 5.4, 5.5, 6.1, 6.2, 6.4, 6.5

        Parameters
        ----------
        user_query:  The user's natural language question.
        history:     Current conversation history.
        backend:     ``"ollama"`` | ``"deepseek"`` | ``"openrouter"``.

        Returns
        -------
        (response_text, updated_history, generated_cypher, cypher_source)
        """
        # Truncate history to last 10 messages (Requirement 6.5).
        truncated = history[-10:] if len(history) > 10 else history

        logger.info("ChatService.query backend=%s query=%s", backend, user_query)

        # ------------------------------------------------------------------
        # Step 1: retrieve context from the graph or vector index
        # ------------------------------------------------------------------
        if rag_mode == "standard_rag":
            import chromadb
            from chromadb.config import Settings
            try:
                client = chromadb.PersistentClient(path="./backend/chroma_db", settings=Settings(allow_reset=True))
                collection = client.get_collection(name="scatterbrain_docs")
                results = collection.query(query_texts=[user_query], n_results=5)
                documents = results.get("documents", [[]])[0]
                context = "\n\n".join(documents)
            except Exception as exc:
                logger.warning("Failed to query ChromaDB: %s", exc)
                context = ""
            generated_cypher = None
            cypher_source = "vector_search"
        else:
            context, generated_cypher, cypher_source = (
                await self._graph_query_service.get_relevant_context(
                    user_query, top_k=5, backend=backend
                )
            )

        # ------------------------------------------------------------------
        # Step 2: generate an answer using the selected backend.
        # ------------------------------------------------------------------
        if backend == "deepseek":
            client = self._get_deepseek_client()
            response_text = await client.generate_answer(
                question=user_query,
                context=context,
                history=truncated,
            )

        elif backend == "openrouter":
            client = self._get_openrouter_client()
            response_text = await client.generate_answer(
                question=user_query,
                context=context,
                history=truncated,
            )

        else:
            # Default: local Ollama (Mistral 7B).
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
        return response_text, updated_history, generated_cypher, cypher_source
