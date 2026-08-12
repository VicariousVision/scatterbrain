"""Neo4j graph service for schema setup, graph summaries, and deletion.

The entity/relationship write operations are now handled by neo4j-graphrag's
``SimpleKGPipeline`` (via ``Neo4jWriter``).  This service retains:
  - connection verification and vector-index creation
  - ``delete_by_document_id`` for stale-data cleanup on re-upload
  - ``get_graph_summary`` for the API summary endpoint

Requirements: 4.4, 4.5
"""

from __future__ import annotations

import logging

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError, ServiceUnavailable

logger = logging.getLogger(__name__)


class GraphServiceError(Exception):
    """Raised when a Neo4j operation fails in an unrecoverable way."""


class GraphService:
    """Manages Neo4j schema setup and document-scoped CRUD for Scatterbrain.

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
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        embedding_dimensions: int = 4096,
    ) -> None:
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
        """Create the ``chunk_embeddings`` vector index used by neo4j-graphrag.

        The ``SimpleKGPipeline`` also creates this index automatically, but
        having it ready at startup avoids a race condition on the first upload.
        Idempotent — uses ``IF NOT EXISTS``.

        Requirements: 4.5
        """
        dims = self._embedding_dimensions
        cypher_vector_index = (
            "CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.embedding) "
            f"OPTIONS {{ indexConfig: {{ `vector.dimensions`: {dims}, "
            f"`vector.similarity_function`: 'cosine' }} }}"
        )
        try:
            with self._driver.session() as session:
                session.run(cypher_vector_index)
                logger.info("Neo4j vector index 'chunk_embeddings' checked/created.")
        except Neo4jError as exc:
            raise GraphServiceError(
                f"Failed to create Neo4j vector index: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all nodes (and their edges) associated with a document.

        Targets ``Document`` and ``Chunk`` nodes written by neo4j-graphrag's
        ``Neo4jWriter``, matched via the ``document_id`` property stored in
        ``document_metadata`` on the ``Document`` node.

        Requirements: 4.4

        Parameters
        ----------
        document_id:
            The UUID of the document whose graph data should be removed.
        """
        cypher = """
            MATCH (doc:Document {document_id: $document_id})
            OPTIONAL MATCH (chunk:Chunk)-[:FROM_DOCUMENT]->(doc)
            OPTIONAL MATCH (entity:__Entity__)-[:FROM_CHUNK]->(chunk)
            DETACH DELETE entity, chunk, doc
        """
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
        """Return node and edge counts for a given document.

        Requirements: 7.5

        Parameters
        ----------
        document_id:
            The UUID stored in the ``Document.document_id`` property.

        Returns
        -------
        dict
            ``{"node_count": int, "edge_count": int}``
        """
        cypher = """
            MATCH (doc:Document {document_id: $document_id})
            OPTIONAL MATCH (chunk:Chunk)-[:FROM_DOCUMENT]->(doc)
            OPTIONAL MATCH (entity:__Entity__)-[:FROM_CHUNK]->(chunk)
            WITH doc,
                 collect(DISTINCT chunk) AS chunks,
                 collect(DISTINCT entity) AS entities
            UNWIND entities AS e
            OPTIONAL MATCH (e)-[entity_rel]->(e2:__Entity__)
            WHERE e2 IN entities
            WITH doc, chunks, entities, count(DISTINCT entity_rel) AS entity_edge_count
            RETURN
                CASE WHEN doc IS NULL THEN 0 ELSE 1 END
                    + size(chunks) + size(entities) AS node_count,
                entity_edge_count
                    + CASE WHEN size(chunks) > 1 THEN size(chunks) - 1 ELSE 0 END
                    + size(chunks) AS edge_count
        """
        try:
            with self._driver.session() as session:
                result = session.run(cypher, document_id=document_id)
                record = result.single()
                if record is None or record["node_count"] == 0:
                    return {"node_count": 0, "edge_count": 0}
                return {
                    "node_count": record["node_count"],
                    "edge_count": record["edge_count"],
                }
        except Neo4jError as exc:
            raise GraphServiceError(
                f"Failed to retrieve graph summary for document_id={document_id}: {exc}"
            ) from exc

    def count_entities_for_document(self, document_id: str) -> int:
        """Return the number of distinct entities linked to a document's chunks.

        Parameters
        ----------
        document_id:
            The UUID stored in the ``Document.document_id`` property.

        Returns
        -------
        int
            Count of ``__Entity__`` nodes extracted from the document's chunks.
        """
        cypher = """
            MATCH (doc:Document {document_id: $document_id})
            MATCH (chunk:Chunk)-[:FROM_DOCUMENT]->(doc)
            MATCH (entity:__Entity__)-[:FROM_CHUNK]->(chunk)
            RETURN count(DISTINCT entity) AS entity_count
        """
        try:
            with self._driver.session() as session:
                result = session.run(cypher, document_id=document_id)
                record = result.single()
                if record is None:
                    return 0
                return int(record["entity_count"])
        except Neo4jError as exc:
            raise GraphServiceError(
                f"Failed to count entities for document_id={document_id}: {exc}"
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
