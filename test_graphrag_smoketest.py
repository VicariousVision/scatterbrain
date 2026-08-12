"""
neo4j-graphrag Smoke Test
=========================
End-to-end test using a hardcoded contract sentence to verify:
  1. SimpleKGPipeline can build a tiny graph via OllamaLLM (Mistral 7B).
  2. Nodes and relationships land in Neo4j correctly.
  3. Text2CypherRetriever can answer a natural-language question.

Connection details are read from the project .env file via backend.config.settings
(NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD / OLLAMA_BASE_URL / OLLAMA_MODEL /
OLLAMA_EMBEDDING_MODEL / OLLAMA_NUM_GPU).

Run from the project root:
    python test_graphrag_smoketest.py

No existing files are touched. To clean up after the test, run in Neo4j Browser:
    MATCH (n) WHERE n.name IN
      ['Alice Corp','John Doe','Employment Agreement','California','confidentiality']
    DETACH DELETE n
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from typing import Any

# Force UTF-8 output on Windows (Python 3.7+) — safe no-op on other platforms
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# ---------------------------------------------------------------------------
# 0. Config (re-use existing settings — no new credentials)
# ---------------------------------------------------------------------------
try:
    from backend.config import settings
except Exception as exc:
    sys.exit(
        f"ERROR: Could not load backend.config.settings.\n"
        f"Make sure you run this from the project root: {exc}"
    )

NEO4J_URI = settings.neo4j_uri
NEO4J_USER = settings.neo4j_username
NEO4J_PASSWORD = settings.neo4j_password
OLLAMA_URL = settings.ollama_base_url
OLLAMA_MODEL = settings.ollama_model             # "mistral"
OLLAMA_EMBED_MODEL = settings.ollama_embedding_model  # "nomic-embed-text"
OLLAMA_NUM_GPU = settings.ollama_num_gpu

# ---------------------------------------------------------------------------
# 1. Test data (hardcoded — no ingestion abstraction)
# ---------------------------------------------------------------------------
TEST_TEXT = (
    "Alice Corp signed an Employment Agreement with John Doe on January 1, 2024. "
    "The agreement is governed by the laws of California. "
    "John Doe is obligated to maintain confidentiality under Section 5."
)

# ---------------------------------------------------------------------------
# 2. Schema (generic contract schema — not the SARB hierarchy)
# ---------------------------------------------------------------------------
NODE_TYPES = [
    {"label": "Organization", "description": "A company or corporate entity, e.g. Alice Corp."},
    {"label": "Person", "description": "An individual person, e.g. John Doe."},
    {
        "label": "Agreement",
        "description": "A legal document or contract, e.g. Employment Agreement.",
        "properties": [
            {"name": "name", "type": "STRING"},
            {"name": "date", "type": "STRING"},
        ],
    },
    {"label": "Jurisdiction", "description": "A legal governing authority or location, e.g. California."},
    {"label": "Obligation", "description": "A duty or requirement imposed on a party, e.g. maintain confidentiality."},
]

RELATIONSHIP_TYPES = [
    {"label": "SIGNED", "description": "A party (Organization or Person) signed the Agreement."},
    {"label": "GOVERNED_BY", "description": "The Agreement is subject to a Jurisdiction's laws."},
    {"label": "OBLIGATED_TO", "description": "A Person is bound to an Obligation under the Agreement."},
    {"label": "RELATES_TO", "description": "An Obligation relates to an Agreement or a specific section."},
]

PATTERNS = [
    ("Organization", "SIGNED", "Agreement"),
    ("Person", "SIGNED", "Agreement"),
    ("Agreement", "GOVERNED_BY", "Jurisdiction"),
    ("Person", "OBLIGATED_TO", "Obligation"),
    ("Obligation", "RELATES_TO", "Agreement"),
]

SCHEMA = {
    "node_types": NODE_TYPES,
    "relationship_types": RELATIONSHIP_TYPES,
    "patterns": PATTERNS,
    "additional_node_types": False,  # strict — only extract the types above
}

# ---------------------------------------------------------------------------
# Neo4j schema string for Text2CypherRetriever (human-readable description)
# ---------------------------------------------------------------------------
NEO4J_SCHEMA_DESC = textwrap.dedent("""\
    Node properties:
      Organization {name: STRING}
      Person {name: STRING}
      Agreement {name: STRING, date: STRING}
      Jurisdiction {name: STRING}
      Obligation {name: STRING}
    The relationships:
      (:Organization)-[:SIGNED]->(:Agreement)
      (:Person)-[:SIGNED]->(:Agreement)
      (:Agreement)-[:GOVERNED_BY]->(:Jurisdiction)
      (:Person)-[:OBLIGATED_TO]->(:Obligation)
      (:Obligation)-[:RELATES_TO]->(:Agreement)
""")

RETRIEVER_EXAMPLES = [
    (
        "USER INPUT: 'Who signed the Employment Agreement?' "
        "QUERY: MATCH (p)-[:SIGNED]->(a:Agreement) RETURN p.name"
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# Import early so we can inherit from it
try:
    from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter, KGWriterModel
except ImportError:
    # If not installed, we'll catch it in main()
    Neo4jWriter = object
    KGWriterModel = object

class NoApocWriter(Neo4jWriter):
    """Drop-in replacement KGWriter that uses pure Cypher MERGE for relationships.

    neo4j_graphrag's default Neo4jWriter calls apoc.merge.relationship which
    requires the APOC plugin. This writer replaces that single method with an
    equivalent pure Cypher query so the smoke test runs without APOC.

    Node writing is inherited unchanged from Neo4jWriter.
    """

    def __init__(self, driver: Any, neo4j_database: Any = None, batch_size: int = 1000) -> None:
        super().__init__(
            driver=driver,
            neo4j_database=neo4j_database,
            batch_size=batch_size,
            clean_db=True,
        )

    # ------------------------------------------------------------------ #
    #  Pure-Cypher relationship upsert (no APOC)                          #
    # ------------------------------------------------------------------ #
    _REL_UPSERT_QUERY = (
        "UNWIND $rows AS row "
        "MATCH (start:__KGBuilder__ {__tmp_internal_id: row.start_node_id}), "
        "      (end:__KGBuilder__   {__tmp_internal_id: row.end_node_id}) "
        # MERGE on the relationship type; properties are set unconditionally
        "CALL { WITH start, end, row "
        "  MERGE (start)-[rel:$(row.type)]->(end) "
        "  SET rel += row.properties "
        "  RETURN rel "
        "} "
        "RETURN elementId(rel)"
    )

    def _upsert_relationships(
        self, rels: list[Any]
    ) -> None:
        """Write relationships using a pure Cypher MERGE (no APOC needed)."""
        rows = [r.model_dump() for r in rels]
        self.driver.execute_query(
            self._REL_UPSERT_QUERY,
            parameters_={"rows": rows},
            database_=self.neo4j_database,
        )

    async def run(self, graph: Any, lexical_graph_config: Any = None) -> KGWriterModel:
        from neo4j_graphrag.experimental.components.types import LexicalGraphConfig
        if lexical_graph_config is None:
            lexical_graph_config = LexicalGraphConfig()
        return await super().run(graph, lexical_graph_config)


# ---------------------------------------------------------------------------
# Main async entrypoint
# ---------------------------------------------------------------------------
async def main() -> None:

    # ------------------------------------------------------------------
    # Phase 0: Import guard — confirm library is importable
    # ------------------------------------------------------------------
    section("Phase 0: Import check")
    try:
        from neo4j import GraphDatabase
        from neo4j_graphrag.embeddings import OllamaEmbeddings
        from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
        from neo4j_graphrag.generation import GraphRAG
        from neo4j_graphrag.llm import OllamaLLM
        from neo4j_graphrag.retrievers import Text2CypherRetriever

        print("[OK] All neo4j-graphrag imports successful")
    except ImportError as exc:
        sys.exit(
            f"IMPORT ERROR: {exc}\n"
            f"Run: pip install \"neo4j-graphrag[experimental,ollama]\""
        )

    # ------------------------------------------------------------------
    # Phase 1: Build shared objects
    # ------------------------------------------------------------------
    section("Phase 1: Setup")

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    print(f"[OK] Neo4j driver created  ->  {NEO4J_URI}")

    # OllamaLLM — num_gpu passed via model_params to stay in CPU-only mode
    llm = OllamaLLM(
        model_name=OLLAMA_MODEL,
        model_params={
            "options": {"num_gpu": OLLAMA_NUM_GPU},
        },
        host=OLLAMA_URL,
    )
    print(f"[OK] OllamaLLM created     ->  model={OLLAMA_MODEL}  host={OLLAMA_URL}")

    # OllamaEmbeddings (used by SimpleKGPipeline for chunk embeddings)
    embedder = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        host=OLLAMA_URL,
    )
    print(f"[OK] OllamaEmbeddings created  ->  model={OLLAMA_EMBED_MODEL}")

    # ------------------------------------------------------------------
    # Phase 2: Build the Knowledge Graph
    # ------------------------------------------------------------------
    section("Phase 2: KG Build (SimpleKGPipeline)")
    
    print("Cleaning up previous smoke test nodes...")
    with driver.session() as session:
        session.run("MATCH (n:__KGBuilder__) DETACH DELETE n")

    print(f"Input text:\n  {TEST_TEXT}\n")

    custom_writer = NoApocWriter(driver=driver)

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        schema=SCHEMA,
        kg_writer=custom_writer,
        from_file=False,               # we are passing raw text
        perform_entity_resolution=False,  # Disabled: requires APOC (apoc.refactor.mergeNodes) which is not installed
        on_error="RAISE",
    )

    print("Running pipeline (this may take 30–120 s on CPU-only Mistral) …")
    result = await kg_builder.run_async(text=TEST_TEXT)
    print(f"[OK] Pipeline result: {result}")

    # ------------------------------------------------------------------
    # Phase 3: Inspect what landed in Neo4j
    # ------------------------------------------------------------------
    section("Phase 3: Cypher dump")

    with driver.session() as session:
        # Nodes
        node_records = session.run(
            "MATCH (n) WHERE n.name IS NOT NULL RETURN labels(n) AS labels, n.name AS name ORDER BY name"
        ).data()
        print(f"\nNodes found ({len(node_records)}):")
        for r in node_records:
            print(f"  [{', '.join(r['labels'])}]  name={r['name']!r}")

        # Relationships
        rel_records = session.run(
            "MATCH (a)-[r]->(b) WHERE a.name IS NOT NULL AND b.name IS NOT NULL "
            "RETURN a.name AS from, type(r) AS rel, b.name AS to ORDER BY from"
        ).data()
        print(f"\nRelationships found ({len(rel_records)}):")
        for r in rel_records:
            print(f"  ({r['from']!r}) -[{r['rel']}]-> ({r['to']!r})")

    print(
        "\nTip: to clean up after this test, run in Neo4j Browser:\n"
        "  MATCH (n) WHERE n.name IN\n"
        "    ['Alice Corp','John Doe','Employment Agreement','California','confidentiality']\n"
        "  DETACH DELETE n"
    )

    # ------------------------------------------------------------------
    # Phase 4: Text2Cypher retrieval
    # ------------------------------------------------------------------
    section("Phase 4: Text2Cypher retrieval")

    retriever = Text2CypherRetriever(
        driver=driver,
        llm=llm,
        neo4j_schema=NEO4J_SCHEMA_DESC,
        examples=RETRIEVER_EXAMPLES,
    )

    # Wrap in GraphRAG to get a natural-language answer
    rag = GraphRAG(retriever=retriever, llm=llm)

    question = "Who signed the agreement with Alice Corp?"
    print(f"Question: {question!r}")
    print("Querying (may take 30–60 s on CPU-only Mistral) ...")

    try:
        response = rag.search(query_text=question, return_context=True)
        print(f"[OK] Answer:  {response.answer}")
        if hasattr(response, "retriever_result") and response.retriever_result:
            print(f"\nRetriever context:")
            for item in response.retriever_result.items:
                print(f"  {item.content}")
    except Exception as exc:
        print(f"[WARN] Retrieval error (graph may be empty or Cypher generation failed): {exc}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    driver.close()
    print("\n[OK] Driver closed. Smoke test complete.")


if __name__ == "__main__":
    asyncio.run(main())
