"""Graph query service using neo4j-graphrag VectorCypherRetriever.

Replaces the hand-rolled keyword + 2-hop traversal with a proper
VectorCypherRetriever that combines vector similarity search with graph
traversal to retrieve richer context for chat queries.

Requirements: 6.2, 6.3, 6.4
"""

from __future__ import annotations

import logging

from neo4j import GraphDatabase, Driver
from neo4j_graphrag.retrievers import VectorCypherRetriever

from backend.services.ollama_adapters import OllamaEmbedderAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retrieval query
# ---------------------------------------------------------------------------
# After the vector search finds the most relevant Chunk nodes, this Cypher
# traversal walks from each chunk to any entities it mentions, then one hop
# further to collect neighbouring entities and their relationships — giving
# the LLM both raw text context and structured graph triples.

_RETRIEVAL_QUERY = """
// node = matched Chunk from vector index
OPTIONAL MATCH (node)-[:PART_OF_CHUNK|PART_OF_DOCUMENT*1..2]->(doc)
OPTIONAL MATCH (node)-[:MENTIONS]->(e:__Entity__)
OPTIONAL MATCH (e)-[r]->(neighbour:__Entity__)
RETURN
    node.text                                   AS chunk_text,
    score                                       AS similarity_score,
    doc.path                                    AS source_document,
    collect(DISTINCT e.name + ' [' + labels(e)[1] + ']')   AS entities,
    collect(DISTINCT e.name + ' -[' + type(r) + ']-> ' + neighbour.name) AS triples
"""


class GraphQueryService:
    """Retrieves relevant context for chat queries using neo4j-graphrag.

    Wraps a ``VectorCypherRetriever`` that performs cosine-similarity search
    over the ``chunk_embeddings`` vector index and then traverses the graph
    to gather entity/relationship context.

    Parameters
    ----------
    uri:
        Bolt or Neo4j URI.
    username:
        Neo4j username.
    password:
        Neo4j password.
    embedder:
        An ``OllamaEmbedderAdapter`` used to embed query text.
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        embedder: OllamaEmbedderAdapter,
    ) -> None:
        self._driver: Driver = GraphDatabase.driver(uri, auth=(username, password))
        self._retriever = VectorCypherRetriever(
            driver=self._driver,
            index_name="chunk_embeddings",
            retrieval_query=_RETRIEVAL_QUERY,
            embedder=embedder,
        )

    async def get_relevant_context(self, query: str, top_k: int = 5) -> str:
        """Return formatted context for a user query.

        Runs the VectorCypherRetriever synchronously (it is a sync API) and
        formats the results into a string suitable for inclusion in an LLM
        prompt.

        Parameters
        ----------
        query:
            The user's natural language question.
        top_k:
            Number of similar chunks to retrieve.

        Returns
        -------
        str
            Formatted context string or a "no data found" fallback.
        """
        try:
            result = self._retriever.search(query_text=query, top_k=top_k)
        except Exception as exc:
            logger.error("VectorCypherRetriever.search failed: %s", exc)
            return "No relevant graph data found for this query."

        if not result.items:
            return "No relevant graph data found for this query."

        sections: list[str] = []
        for item in result.items:
            # item.content is the formatted record string from neo4j-graphrag
            sections.append(item.content)

        return "\n\n---\n\n".join(sections)

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        self._driver.close()

    def __enter__(self) -> "GraphQueryService":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
