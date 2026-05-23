"""Neo4j graph service for storing and querying the legal knowledge graph.

Wraps the ``neo4j`` Python driver and exposes high-level CRUD operations used
by the document processing pipeline and the graph-RAG query path.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

from __future__ import annotations

import logging
from typing import List

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from backend.models.entities import Entity, Relationship

logger = logging.getLogger(__name__)


class GraphServiceError(Exception):
    """Raised when a Neo4j operation fails in an unrecoverable way."""


class GraphService:
    """Manages all Neo4j read/write operations for the knowledge graph.

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

    # ------------------------------------------------------------------
    # Connection / schema
    # ------------------------------------------------------------------

    def verify_connection(self) -> None:
        """Verify that the Neo4j instance is reachable.

        Raises
        ------
        GraphServiceError
            If the driver cannot reach the database.
        """
        try:
            self._driver.verify_connectivity()
            logger.info("Neo4j connectivity verified.")
        except ServiceUnavailable as exc:
            raise GraphServiceError(
                f"Cannot connect to Neo4j at {self._uri}: {exc}"
            ) from exc
        except Exception as exc:
            raise GraphServiceError(
                f"Unexpected error verifying Neo4j connection: {exc}"
            ) from exc

    def create_constraints(self) -> None:
        """Create the uniqueness constraint on (Entity.name, Entity.type).

        Idempotent — uses ``IF NOT EXISTS`` so it is safe to call on every
        startup.

        Requirements: 4.5
        """
        cypher = (
            "CREATE CONSTRAINT entity_name_type_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE"
        )
        try:
            with self._driver.session() as session:
                session.run(cypher)
            logger.info("Neo4j uniqueness constraint applied.")
        except Neo4jError as exc:
            raise GraphServiceError(
                f"Failed to create Neo4j constraints: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def store_entities(self, entities: List[Entity]) -> None:
        """Persist a list of entities to Neo4j using MERGE on name+type.

        New nodes are created with all four properties.  Existing nodes
        (matched by name+type) are left unchanged so that cross-document
        deduplication is preserved.

        Requirements: 4.1, 4.3

        Parameters
        ----------
        entities:
            Entities to upsert.
        """
        if not entities:
            return

        cypher = (
            "MERGE (e:Entity {name: $name, type: $type}) "
            "ON CREATE SET e.id = $id, e.document_id = $document_id "
            "ON MATCH SET e.document_id = e.document_id "
            "RETURN e"
        )
        try:
            with self._driver.session() as session:
                for entity in entities:
                    session.run(
                        cypher,
                        name=entity.name,
                        type=entity.type,
                        id=entity.id,
                        document_id=entity.document_id,
                    )
            logger.debug("Stored %d entities.", len(entities))
        except Neo4jError as exc:
            raise GraphServiceError(
                f"Failed to store entities: {exc}"
            ) from exc

    def store_relationships(self, relationships: List[Relationship]) -> None:
        """Persist a list of relationships to Neo4j using MERGE.

        Each relationship is merged on (source_name, target_name, rel_type,
        document_id).  Source and target nodes must already exist; if either
        is missing the relationship is silently skipped (MATCH returns nothing).

        Requirements: 4.2

        Parameters
        ----------
        relationships:
            Relationships to upsert.
        """
        if not relationships:
            return

        cypher = (
            "MATCH (source:Entity {name: $source_name}) "
            "MATCH (target:Entity {name: $target_name}) "
            "MERGE (source)-[r:RELATIONSHIP {type: $rel_type, document_id: $document_id}]->(target) "
            "RETURN r"
        )
        try:
            with self._driver.session() as session:
                for rel in relationships:
                    session.run(
                        cypher,
                        source_name=rel.source_entity,
                        target_name=rel.target_entity,
                        rel_type=rel.relationship_type,
                        document_id=rel.document_id,
                    )
            logger.debug("Stored %d relationships.", len(relationships))
        except Neo4jError as exc:
            raise GraphServiceError(
                f"Failed to store relationships: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all nodes (and their edges) associated with a document.

        Used before re-processing a re-uploaded document so that stale data
        is removed before fresh data is written.

        Requirements: 4.4

        Parameters
        ----------
        document_id:
            The UUID of the document whose graph data should be removed.
        """
        cypher = (
            "MATCH (e:Entity {document_id: $document_id}) "
            "DETACH DELETE e"
        )
        try:
            with self._driver.session() as session:
                result = session.run(cypher, document_id=document_id)
                summary = result.consume()
                logger.info(
                    "Deleted %d nodes for document_id=%s.",
                    summary.counters.nodes_deleted,
                    document_id,
                )
        except Neo4jError as exc:
            raise GraphServiceError(
                f"Failed to delete graph data for document_id={document_id}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_graph_summary(self, document_id: str) -> dict:
        """Return the node and edge counts for a given document.

        Requirements: 7.5

        Parameters
        ----------
        document_id:
            The UUID of the document to summarise.

        Returns
        -------
        dict
            ``{"node_count": int, "edge_count": int}``
        """
        cypher = (
            "MATCH (e:Entity {document_id: $document_id}) "
            "OPTIONAL MATCH (e)-[r {document_id: $document_id}]-() "
            "RETURN count(DISTINCT e) AS node_count, count(DISTINCT r) AS edge_count"
        )
        try:
            with self._driver.session() as session:
                result = session.run(cypher, document_id=document_id)
                record = result.single()
                if record is None:
                    return {"node_count": 0, "edge_count": 0}
                return {
                    "node_count": record["node_count"],
                    "edge_count": record["edge_count"],
                }
        except Neo4jError as exc:
            raise GraphServiceError(
                f"Failed to retrieve graph summary for document_id={document_id}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying Neo4j driver and release resources."""
        self._driver.close()

    def __enter__(self) -> "GraphService":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
