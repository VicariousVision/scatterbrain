"""Neo4j schema setup for the Scatterbrain SARB manual graph.

Creates constraints and indexes for the domain-specific schema.
No vector index, no embedding property — this pipeline uses Cypher traversal.

Node labels:
  Manual      — singleton root
  Provision   — any numbered node at any depth; path is the unique key
  Definition  — A.1 glossary entries; term is the unique key
  Limit       — extracted monetary/quantitative limits
  PartyRole   — closed set, hand-seeded
  Entity      — named institutions (Authorised Dealers, ADLAs)
  LegalInstrument — referenced external Acts/Regulations

Run create_schema() once at startup (idempotent via IF NOT EXISTS).
"""

from __future__ import annotations

import logging

from neo4j import Driver
from neo4j.exceptions import Neo4jError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constraint / index DDL
# ---------------------------------------------------------------------------

_CONSTRAINTS = [
    # Uniqueness constraints (also create an implicit index)
    "CREATE CONSTRAINT provision_path IF NOT EXISTS FOR (p:Provision) REQUIRE p.path IS UNIQUE",
    "CREATE CONSTRAINT definition_term IF NOT EXISTS FOR (d:Definition) REQUIRE d.term IS UNIQUE",
    "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT manual_name IF NOT EXISTS FOR (m:Manual) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT legal_instrument_name IF NOT EXISTS FOR (l:LegalInstrument) REQUIRE l.name IS UNIQUE",
    "CREATE CONSTRAINT party_role_name IF NOT EXISTS FOR (r:PartyRole) REQUIRE r.name IS UNIQUE",
]

_INDEXES = [
    "CREATE INDEX provision_level IF NOT EXISTS FOR (p:Provision) ON (p.level)",
]

# ---------------------------------------------------------------------------
# Closed-set party roles — seeded once at startup
# ---------------------------------------------------------------------------

_PARTY_ROLES = [
    "Resident",
    "NonResident",
    "Minor",
    "Student",
    "Immigrant",
    "ProspectiveImmigrant",
    "ForeignNational",
    "CMAResident",
    "AffectedPerson",
    "Spouse",
    "Other",
]


class GraphSchemaService:
    """Applies the domain schema to a Neo4j database.

    Parameters
    ----------
    driver:
        An open synchronous Neo4j driver.
    """

    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    def create_schema(self) -> None:
        """Create all constraints, indexes, and seed PartyRole nodes.

        Idempotent — safe to call on every startup.
        """
        self._apply_ddl()
        self._seed_party_roles()

    def _apply_ddl(self) -> None:
        with self._driver.session() as session:
            for stmt in _CONSTRAINTS + _INDEXES:
                try:
                    session.run(stmt)
                    logger.debug("Applied: %s", stmt[:60])
                except Neo4jError as exc:
                    # ConstraintAlreadyExists / EquivalentSchemaRuleAlreadyExists
                    # are safe to ignore — IF NOT EXISTS should handle them but
                    # older Neo4j versions still raise.
                    if "already exists" in str(exc).lower() or "equivalent" in str(exc).lower():
                        logger.debug("Schema already exists (skipped): %s", stmt[:60])
                    else:
                        logger.error("Failed to apply DDL: %s — %s", stmt[:60], exc)
                        raise
        logger.info("GraphSchemaService: constraints and indexes applied.")

    def _seed_party_roles(self) -> None:
        """MERGE PartyRole nodes for the closed set of roles."""
        cypher = "MERGE (:PartyRole {name: $name})"
        with self._driver.session() as session:
            for role in _PARTY_ROLES:
                session.run(cypher, name=role)
        logger.info("GraphSchemaService: %d PartyRole nodes seeded.", len(_PARTY_ROLES))

    def delete_manual_graph(self, manual_name: str) -> int:
        """Delete all Provision/Definition nodes for a given manual.

        Returns the number of nodes deleted.  Used before re-ingesting.
        """
        cypher = """
            MATCH (m:Manual {name: $name})
            OPTIONAL MATCH (m)-[:HAS_TOP_SECTION*0..]->(p)
            WITH m, collect(DISTINCT p) AS provisions
            UNWIND provisions AS node
            DETACH DELETE node
            WITH m
            DETACH DELETE m
            RETURN count(*) AS deleted
        """
        with self._driver.session() as session:
            result = session.run(cypher, name=manual_name)
            record = result.single()
            deleted = int(record["deleted"]) if record else 0
        logger.info(
            "Deleted %d nodes for manual '%s'.", deleted, manual_name
        )
        return deleted
