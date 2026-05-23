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

    async def get_relevant_context(self, query: str) -> str:
        """Retrieve relevant graph context for a user query.

        Tokenizes the query, performs case-insensitive substring matching
        against entity names, executes 2-hop traversal for each matched
        entity, and formats the results as triples.

        Requirements: 6.2, 6.3, 6.4

        Parameters
        ----------
        query:
            The user's natural language query.

        Returns
        -------
        str
            Formatted triples in the form ``(name) -[TYPE]-> (name)``, or
            ``"No relevant graph data found for this query."`` if no matches.
        """
        # Tokenize query into terms (simple whitespace split)
        terms = query.lower().split()
        if not terms:
            return "No relevant graph data found for this query."

        # Collect all paths from 2-hop traversal for each term
        all_paths = []
        for term in terms:
            paths = self._execute_2hop_traversal(term)
            all_paths.extend(paths)

        # Format paths as triples
        return self._format_triples(all_paths)

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
