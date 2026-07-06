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
    embedding_dimensions:
        Dimensionality of the embedding vectors for the vector index.
        Defaults to ``4096``.
    """

    def __init__(self, uri: str, username: str, password: str, embedding_dimensions: int = 4096) -> None:
        self._uri = uri
        self._username = username
        self._password = password
        self._embedding_dimensions = embedding_dimensions
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
        """Create the uniqueness constraint on (Entity.name, Entity.type) and the vector index.

        Idempotent — uses ``IF NOT EXISTS`` so it is safe to call on every
        startup.

        Requirements: 4.5
        """
        cypher_constraint = (
            "CREATE CONSTRAINT entity_name_type_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE"
        )
        dims = self._embedding_dimensions
        cypher_vector_index = (
            "CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.embedding) "
            f"OPTIONS {{ indexConfig: {{ `vector.dimensions`: {dims}, `vector.similarity_function`: 'cosine' }} }}"
        )
        try:
            with self._driver.session() as session:
                session.run(cypher_constraint)
                logger.info("Neo4j uniqueness constraint applied.")
                session.run(cypher_vector_index)
                logger.info("Neo4j vector index 'chunk_embeddings' checked/created.")
        except Neo4jError as exc:
            raise GraphServiceError(
                f"Failed to create Neo4j constraints or vector index: {exc}"
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

    def store_chunks(self, chunks: List[dict]) -> None:
        """Persist a list of document chunks to Neo4j.

        Parameters
        ----------
        chunks:
            List of dicts, each with keys: ``"id"``, ``"text"``, ``"document_id"``, ``"embedding"``.
        """
        if not chunks:
            return

        cypher = (
            "MERGE (c:Chunk {id: $id}) "
            "SET c.text = $text, c.document_id = $document_id, c.embedding = $embedding"
        )
        try:
            with self._driver.session() as session:
                for chunk in chunks:
                    session.run(
                        cypher,
                        id=chunk["id"],
                        text=chunk["text"],
                        document_id=chunk["document_id"],
                        embedding=chunk["embedding"],
                    )
            logger.debug("Stored %d chunks.", len(chunks))
        except Neo4jError as exc:
            raise GraphServiceError(
                f"Failed to store chunks: {exc}"
            ) from exc

    def link_chunks_to_entities(self, document_id: str) -> None:
        """Link Chunk nodes to Entity nodes based on text overlap within a document.

        Matches (c:Chunk) and (e:Entity) for the given document_id, and creates a
        [:MENTIONS] relationship from the chunk to the entity if the chunk's text
        contains the entity's name (case-insensitive).
        """
        cypher = (
            "MATCH (c:Chunk {document_id: $document_id}) "
            "MATCH (e:Entity {document_id: $document_id}) "
            "WHERE toLower(c.text) CONTAINS toLower(e.name) "
            "MERGE (c)-[:MENTIONS]->(e)"
        )
        try:
            with self._driver.session() as session:
                session.run(cypher, document_id=document_id)
            logger.info("Linked chunks to entities for document_id=%s", document_id)
        except Neo4jError as exc:
            raise GraphServiceError(
                f"Failed to link chunks to entities: {exc}"
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
        cypher_chunks = (
            "MATCH (c:Chunk {document_id: $document_id}) "
            "DETACH DELETE c"
        )
        cypher_entities = (
            "MATCH (e:Entity {document_id: $document_id}) "
            "DETACH DELETE e"
        )
        try:
            with self._driver.session() as session:
                result_c = session.run(cypher_chunks, document_id=document_id)
                summary_c = result_c.consume()
                result_e = session.run(cypher_entities, document_id=document_id)
                summary_e = result_e.consume()
                logger.info(
                    "Deleted %d nodes (entities: %d, chunks: %d) for document_id=%s.",
                    summary_c.counters.nodes_deleted + summary_e.counters.nodes_deleted,
                    summary_e.counters.nodes_deleted,
                    summary_c.counters.nodes_deleted,
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
