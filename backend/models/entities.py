"""Pydantic models for extracted entities and relationships.

These models represent the structured output of the Entity Extractor and are
used when storing data to and reading data from the Neo4j Knowledge Graph.

Valid entity types: Person, Organization, Contract, Clause, Date, Jurisdiction, Obligation
Requirements: 3.4, 3.6, 4.1
"""

from pydantic import BaseModel


class Entity(BaseModel):
    """A named entity extracted from a legal document.

    The ``id`` field is a UUID4 string generated at extraction time.
    The ``type`` field must be one of the seven legal entity types defined in
    the extraction prompt: Person, Organization, Contract, Clause, Date,
    Jurisdiction, Obligation.

    Requirements: 3.4, 3.6, 4.1
    """

    id: str  # UUID4
    name: str
    type: str  # Person | Organization | Contract | Clause | Date | Jurisdiction | Obligation
    document_id: str


class Relationship(BaseModel):
    """A directed relationship between two entities in a legal document.

    Both ``source_entity`` and ``target_entity`` refer to entity names (not
    IDs) so they can be matched against existing nodes in Neo4j via MERGE.

    Requirements: 3.3, 3.4, 3.6
    """

    source_entity: str
    relationship_type: str
    target_entity: str
    document_id: str


class ExtractionResult(BaseModel):
    """The complete output of a single entity-extraction LLM call.

    Wraps the lists of entities and relationships returned after parsing the
    LLM's JSON response.  Both lists may be empty if the document contains no
    recognisable legal entities.

    Requirements: 3.4
    """

    entities: list[Entity]
    relationships: list[Relationship]
