"""Neo4j graph service: connection verification, summary queries, and deletion.

The entity/relationship write operations are handled by Neo4jLoader in
extraction_service.py.  This service provides:
  - Connection verification at startup
  - delete_by_document_id for re-upload cleanup (kept for API compatibility)
  - get_graph_summary for the /documents/{id}/graph-summary endpoint
  - count_provisions_for_manual used by DocumentService as a sanity check

No vector index creation here — this pipeline does not use embeddings.

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
    """Manages Neo4j connectivity and document-scoped CRUD for Scatterbrain.

    Parameters
    ----------
    uri:      Bolt or Neo4j URI, e.g. ``bolt://localhost:7687``.
    username: Neo4j username.
    password: Neo4j password.
    embedding_dimensions:
        Kept for API compatibility but no longer used — vector index is not
        created by this service.
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        embedding_dimensions: int = 768,
    ) -> None:
        self._uri = uri
        self._username = username
        self._password = password
        # embedding_dimensions is retained so callers that pass it (main.py)
        # don't need to change their signature.
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
        """No-op: schema setup is now delegated to GraphSchemaService.

        Kept so main.py's startup sequence does not need to change.
        GraphSchemaService.create_schema() is called inside DocumentService's
        ingestion pipeline (idempotent on each upload).
        """
        logger.info(
            "GraphService.create_constraints: schema managed by GraphSchemaService "
            "(called per-upload). No vector index created — pipeline uses Cypher."
        )

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all Provision/Definition nodes associated with a Manual.

        For the new schema, document_id is not stored on graph nodes (the
        manual_name derived from the filename is the identifier).  This method
        is kept for API compatibility with the router; it is a no-op when
        called before a manual has been ingested under the new schema.

        For a full re-ingest cleanup, use GraphSchemaService.delete_manual_graph.

        Requirements: 4.4
        """
        # In the new schema we don't tag every Provision with a document_id,
        # so we log and return.  The upload flow in DocumentService handles
        # stale-data cleanup by deleting old document records from the in-memory
        # store; the actual graph nodes are overwritten via MERGE (idempotent).
        logger.debug(
            "delete_by_document_id called for document_id=%s — "
            "new schema uses MERGE (idempotent), stale node cleanup skipped.",
            document_id,
        )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_graph_summary(self, document_id: str) -> dict:
        """Return Provision and Definition counts across the whole graph.

        The new schema does not tag nodes with document_id, so we return
        global counts.  The document_id parameter is kept for API compat.

        Returns
        -------
        dict
            ``{"node_count": int, "edge_count": int}``
        """
        cypher = """
            MATCH (p:Provision) WITH count(p) AS pCount
            MATCH (d:Definition) WITH pCount, count(d) AS dCount
            MATCH ()-[r]->() WITH pCount, dCount, count(r) AS eCount
            RETURN pCount + dCount AS node_count, eCount AS edge_count
        """
        try:
            with self._driver.session() as session:
                result = session.run(cypher)
                record = result.single()
                if record is None:
                    return {"node_count": 0, "edge_count": 0}
                return {
                    "node_count": record["node_count"],
                    "edge_count": record["edge_count"],
                }
        except Neo4jError as exc:
            raise GraphServiceError(
                f"Failed to retrieve graph summary: {exc}"
            ) from exc

    def count_entities_for_document(self, document_id: str) -> int:
        """Kept for API compat.  Returns total Provision count instead."""
        try:
            with self._driver.session() as session:
                result = session.run("MATCH (p:Provision) RETURN count(p) AS n")
                record = result.single()
                return int(record["n"]) if record else 0
        except Neo4jError:
            return 0

    def count_provisions_for_manual(self, manual_name: str) -> int:
        """Return the number of Provision nodes for the given manual.

        Parameters
        ----------
        manual_name:
            The ``name`` property on the Manual root node.
        """
        cypher = """
            MATCH (m:Manual {name: $name})-[:HAS_TOP_SECTION|HAS_CHILD*1..]->(p:Provision)
            RETURN count(DISTINCT p) AS n
        """
        try:
            with self._driver.session() as session:
                result = session.run(cypher, name=manual_name)
                record = result.single()
                return int(record["n"]) if record else 0
        except Neo4jError as exc:
            logger.warning(
                "count_provisions_for_manual('%s') failed: %s", manual_name, exc
            )
            return 0

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
