"""
Property-based tests for the graph service.

**Validates: Requirements 4.1, 4.3**
"""

from __future__ import annotations

import os
import uuid
from typing import Generator

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from backend.models.entities import Entity
from backend.services.graph_service import GraphService


# ---------------------------------------------------------------------------
# Neo4j connection fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def neo4j_connection() -> Generator[dict, None, None]:
    """
    Fixture that checks Neo4j connectivity and provides connection details.
    
    Skips tests if Neo4j is unavailable.
    """
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j")
    
    # Test connectivity
    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        driver.close()
    except ServiceUnavailable:
        pytest.skip(f"Neo4j is not available at {uri}. Skipping integration tests.")
    except Exception as e:
        pytest.skip(f"Cannot connect to Neo4j: {e}. Skipping integration tests.")
    
    yield {"uri": uri, "username": username, "password": password}


@pytest.fixture
def graph_service(neo4j_connection: dict) -> Generator[GraphService, None, None]:
    """
    Fixture that provides a GraphService instance and cleans up test data.
    """
    service = GraphService(
        uri=neo4j_connection["uri"],
        username=neo4j_connection["username"],
        password=neo4j_connection["password"],
    )
    
    yield service
    
    # Cleanup: delete all test nodes (those with document_id starting with 'test-')
    cleanup_cypher = (
        "MATCH (e:Entity) "
        "WHERE e.document_id STARTS WITH 'test-' "
        "DETACH DELETE e"
    )
    try:
        with service._driver.session() as session:
            session.run(cleanup_cypher)
    except Exception:
        pass  # Best effort cleanup
    
    service.close()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Non-empty text for entity names
_entity_name_strategy = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=50,
)

# Valid entity types
_entity_type_strategy = st.sampled_from([
    "Person",
    "Organization",
    "Contract",
    "Clause",
    "Date",
    "Jurisdiction",
    "Obligation",
])

# Test document IDs (prefixed with 'test-' for cleanup)
_test_document_id_strategy = st.builds(
    lambda: f"test-{uuid.uuid4()}",
)


def _entity_strategy(document_id: str | None = None) -> st.SearchStrategy[Entity]:
    """Strategy for generating Entity objects."""
    if document_id is None:
        doc_id_strat = _test_document_id_strategy
    else:
        doc_id_strat = st.just(document_id)
    
    return st.builds(
        Entity,
        id=st.builds(lambda: str(uuid.uuid4())),
        name=_entity_name_strategy,
        type=_entity_type_strategy,
        document_id=doc_id_strat,
    )


# ---------------------------------------------------------------------------
# Property 8: Stored graph nodes have all required properties
# ---------------------------------------------------------------------------


@pytest.mark.integration
@given(
    entities=st.lists(_entity_strategy(), min_size=1, max_size=5),
)
@settings(max_examples=10, deadline=10000)
def test_stored_nodes_have_required_properties(
    graph_service: GraphService,
    entities: list[Entity],
) -> None:
    """
    Property 8: Stored graph nodes have all required properties.

    For any list of entities stored to Neo4j, querying the graph must return
    nodes where every node has non-null id, name, type, and document_id
    properties.

    **Validates: Requirements 4.1**
    """
    # Store entities
    graph_service.store_entities(entities)
    
    # Query Neo4j directly to retrieve stored nodes
    cypher = (
        "MATCH (e:Entity) "
        "WHERE e.document_id STARTS WITH 'test-' "
        "RETURN e.id AS id, e.name AS name, e.type AS type, e.document_id AS document_id"
    )
    
    with graph_service._driver.session() as session:
        result = session.run(cypher)
        records = list(result)
    
    # Assert we got at least as many nodes as we stored (may be more from previous tests)
    assert len(records) >= len(entities), (
        f"Expected at least {len(entities)} nodes, but found {len(records)}"
    )
    
    # Assert every node has all required properties (non-null)
    for record in records:
        assert record["id"] is not None, "Node id must not be null"
        assert record["name"] is not None, "Node name must not be null"
        assert record["type"] is not None, "Node type must not be null"
        assert record["document_id"] is not None, "Node document_id must not be null"
        
        # Additional validation: properties must be non-empty strings
        assert isinstance(record["id"], str) and len(record["id"]) > 0, (
            "Node id must be a non-empty string"
        )
        assert isinstance(record["name"], str) and len(record["name"]) > 0, (
            "Node name must be a non-empty string"
        )
        assert isinstance(record["type"], str) and len(record["type"]) > 0, (
            "Node type must be a non-empty string"
        )
        assert isinstance(record["document_id"], str) and len(record["document_id"]) > 0, (
            "Node document_id must be a non-empty string"
        )


# ---------------------------------------------------------------------------
# Property 9: Entity deduplication
# ---------------------------------------------------------------------------


@pytest.mark.integration
@given(
    name=_entity_name_strategy,
    entity_type=_entity_type_strategy,
)
@settings(max_examples=10, deadline=10000)
def test_entity_deduplication(
    graph_service: GraphService,
    name: str,
    entity_type: str,
) -> None:
    """
    Property 9: Entity deduplication — inserting the same entity twice creates
    exactly one node.

    For any entity with a given name and type, inserting it into the Knowledge
    Graph twice (even with different document_id values) must result in exactly
    one node in the graph with that name+type combination.

    **Validates: Requirements 4.3**
    """
    # Create two entities with the same name and type but different document_ids
    doc_id_1 = f"test-{uuid.uuid4()}"
    doc_id_2 = f"test-{uuid.uuid4()}"
    
    entity_1 = Entity(
        id=str(uuid.uuid4()),
        name=name,
        type=entity_type,
        document_id=doc_id_1,
    )
    
    entity_2 = Entity(
        id=str(uuid.uuid4()),
        name=name,
        type=entity_type,
        document_id=doc_id_2,
    )
    
    # Store both entities
    graph_service.store_entities([entity_1])
    graph_service.store_entities([entity_2])
    
    # Query Neo4j to count nodes with this name+type combination
    cypher = (
        "MATCH (e:Entity {name: $name, type: $type}) "
        "RETURN count(e) AS node_count"
    )
    
    with graph_service._driver.session() as session:
        result = session.run(cypher, name=name, type=entity_type)
        record = result.single()
        node_count = record["node_count"]
    
    # Assert exactly one node exists (MERGE deduplication worked)
    assert node_count == 1, (
        f"Expected exactly 1 node with name={name!r} and type={entity_type!r}, "
        f"but found {node_count} nodes. MERGE deduplication failed."
    )
