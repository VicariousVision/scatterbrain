"""Graph query service for retrieving relevant subgraph context for chat queries.

Implements the graph-RAG retrieval pattern: tokenizes user queries, matches
entities by name, and performs 2-hop traversal to gather relevant context.

Requirements: 6.2, 6.3, 6.4
"""

from __future__ import annotations

import logging
from typing import List

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError

logger = logging.getLogger(__name__)


class GraphQueryService:
    """Retrieves relevant subgraph context for user queries.

    Parameters
    ----------
    uri:
        Bolt or Neo4j URI, e.g. ``bolt://localhost:7687``.
    username:
        Neo4j username.
    password:
        Neo4j password.
    """

    def __init__(self, uri: str, username: str, password: str) -> None:
        self._uri = uri
        self._username = username
        self._password = password
        self._driver: Driver = GraphDatabase.driver(uri, auth=(username, password))

    async def get_relevant_context(self, query: str, query_vector: list[float] | None = None) -> str:
        """Retrieve relevant graph and vector context for a user query.

        Tokenizes the query, performs case-insensitive substring matching
        against entity names, executes 2-hop traversal for each matched
        entity, performs vector search for relevant text chunks, and formats
        the results.

        Requirements: 6.2, 6.3, 6.4

        Parameters
        ----------
        query:
            The user's natural language query.
        query_vector:
            Optional embedding vector of the query.

        Returns
        -------
        str
            Formatted triples in the form ``(name) -[TYPE]-> (name)`` and text chunks,
            or ``"No relevant graph data found for this query."`` if no matches.
        """
        # Tokenize query into terms (simple whitespace split)
        terms = query.lower().split()
        if not terms:
            return "No relevant graph data found for this query."

        # Collect all paths from 2-hop traversal for each term (keyword matching)
        all_paths = []
        for term in terms:
            paths = self._execute_2hop_traversal(term)
            all_paths.extend(paths)

        chunks_context = ""
        if query_vector is not None:
            # Retrieve semantically similar chunks
            chunks = self._retrieve_similar_chunks(query_vector, limit=5)
            if chunks:
                chunks_context = "\nTEXT CHUNKS CONTEXT:\n" + "\n".join(
                    f"- {c['text']}" for c in chunks
                )

            # Retrieve structural paths starting from entities mentioned in these chunks
            vector_paths = self._execute_vector_paths_traversal(query_vector, limit=5)
            all_paths.extend(vector_paths)

        # Format paths as triples
        graph_triples = self._format_triples(all_paths)

        if query_vector is not None:
            return f"GRAPH CONTEXT:\n{graph_triples}\n{chunks_context}"
        else:
            return graph_triples

    def _retrieve_similar_chunks(self, query_vector: list[float], limit: int = 5) -> list[dict]:
        """Query the vector index for similar text chunks.

        Parameters
        ----------
        query_vector:
            The embedding vector of the query.
        limit:
            Maximum number of chunks to return.

        Returns
        -------
        list[dict]
            List of dicts with 'text', 'document_id', and 'score'.
        """
        cypher = """
        CALL db.index.vector.queryNodes('chunk_embeddings', $limit, $query_vector)
        YIELD node, score
        RETURN node.text AS text, node.document_id AS document_id, score
        """
        try:
            with self._driver.session() as session:
                result = session.run(cypher, query_vector=query_vector, limit=limit)
                return [
                    {
                        "text": record["text"],
                        "document_id": record["document_id"],
                        "score": record["score"],
                    }
                    for record in result
                ]
        except Neo4jError as exc:
            logger.error("Failed to query vector index: %s", exc)
            return []

    def _execute_vector_paths_traversal(self, query_vector: list[float], limit: int = 5) -> list:
        """Retrieve paths within 1-hop of entities mentioned in semantically relevant chunks.

        Parameters
        ----------
        query_vector:
            The embedding vector of the query.
        limit:
            Maximum number of chunks to traverse from.

        Returns
        -------
        list
            List of Neo4j path objects.
        """
        cypher = """
        CALL db.index.vector.queryNodes('chunk_embeddings', $limit, $query_vector)
        YIELD node AS chunk
        MATCH (chunk)-[:MENTIONS]->(e:Entity)
        MATCH path = (e)-[*1]-(neighbor)
        RETURN path
        """
        try:
            with self._driver.session() as session:
                result = session.run(cypher, query_vector=query_vector, limit=limit)
                return [record["path"] for record in result]
        except Neo4jError as exc:
            logger.error("Failed to execute vector paths traversal: %s", exc)
            return []

    def _execute_2hop_traversal(self, term: str) -> List[dict]:
        """Execute 2-hop traversal query for entities matching the given term.

        Parameters
        ----------
        term:
            A single query term (already lowercased).

        Returns
        -------
        List[dict]
            List of path records containing nodes and relationships.
        """
        cypher = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS $term
        MATCH path = (e)-[*1..2]-(neighbor)
        RETURN path
        """
        try:
            with self._driver.session() as session:
                result = session.run(cypher, term=term)
                return [record["path"] for record in result]
        except Neo4jError as exc:
            logger.error("Failed to execute 2-hop traversal for term '%s': %s", term, exc)
            return []

    def _format_triples(self, paths: List) -> str:
        """Format graph paths as triple strings.

        Requirements: 6.3

        Parameters
        ----------
        paths:
            List of Neo4j path objects containing nodes and relationships.

        Returns
        -------
        str
            Formatted triples, one per line, or the "no data found" message.
        """
        if not paths:
            return "No relevant graph data found for this query."

        triples = set()  # Use set to deduplicate triples
        for path in paths:
            # Extract relationships from the path
            relationships = path.relationships
            for rel in relationships:
                # Get start and end node names
                start_node = rel.start_node
                end_node = rel.end_node
                start_name = start_node.get("name", "Unknown")
                end_name = end_node.get("name", "Unknown")
                # The relationship type is stored in the 'type' property
                # because all relationships have the label 'RELATIONSHIP'
                rel_type = rel.get("type", "UNKNOWN")

                # Format as triple
                triple = f"({start_name}) -[{rel_type}]-> ({end_name})"
                triples.add(triple)

        if not triples:
            return "No relevant graph data found for this query."

        return "\n".join(sorted(triples))  # Sort for consistent output

    def close(self) -> None:
        """Close the underlying Neo4j driver and release resources."""
        self._driver.close()

    def __enter__(self) -> "GraphQueryService":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
