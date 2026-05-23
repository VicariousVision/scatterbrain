# Scatterbrain

Scatterbrain is a legal document intelligence platform that combines document parsing, knowledge graph extraction, and Graph-RAG-powered conversational querying.

Users can upload legal documents (`PDF`, `DOCX`, `TXT`), automatically extract entities and relationships into a Neo4j knowledge graph, and query the graph using a conversational interface backed by a locally-hosted LLM running through Ollama.

The system is composed of:

- A **Streamlit frontend** for document upload and chat interaction.
- A **FastAPI backend** for document processing, entity extraction, graph storage, and LLM orchestration.
- A **Neo4j graph database** for storing entities and relationships.
- A **local Ollama-hosted LLM** for extraction and grounded question answering.

---

# Features

## Document Upload

- Upload `.pdf`, `.docx`, and `.txt` legal documents.
- Background document processing workflow.
- Upload progress and status tracking.
- Document management dashboard.

## Document Parsing

- PDF text extraction.
- DOCX paragraph and table extraction.
- TXT ingestion.
- Preservation of paragraph structure.

## Entity & Relationship Extraction

Automatic extraction of:

- Persons
- Organizations
- Contracts
- Clauses
- Dates
- Jurisdictions
- Obligations

Relationships are extracted as directed graph edges.

Example:

```text
(Alice Corp) -[SIGNED]-> (Employment Agreement)
```

## Knowledge Graph Storage

- Neo4j-backed graph storage.
- Entity deduplication using composite keys.
- Cross-document relationship discovery.
- Graph summaries per document.

## Conversational Graph-RAG Chat

- Natural language querying over uploaded legal documents.
- Responses grounded in graph context.
- Conversation history support.
- Multi-hop graph retrieval.

## System Health & Configuration

- Environment variable configuration.
- Health check endpoint.
- Startup dependency verification.

---

# Architecture

```text
                +----------------------+
                |     Streamlit UI     |
                |----------------------|
                | Upload Page          |
                | Chat Interface       |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |    FastAPI Backend   |
                |----------------------|
                | Upload API           |
                | Parsing Engine       |
                | Entity Extractor     |
                | Graph-RAG Orchestrator|
                +----------+-----------+
                           |
          +----------------+----------------+
          |                                 |
          v                                 v
+-------------------+           +-------------------+
|      Neo4j        |           |      Ollama       |
| Knowledge Graph   |           | Local LLM Runtime |
+-------------------+           +-------------------+
```

---

# Tech Stack

## Frontend

- Streamlit

## Backend

- FastAPI
- Uvicorn

## AI / NLP

- Ollama
- Local LLM

## Database

- Neo4j

## Parsing Libraries

- PyPDF2 / pdfplumber
- python-docx

---

# Project Structure

```text
scatterbrain/
│
├── backend/
│   ├── api/
│   ├── services/
│   │   ├── parser/
│   │   ├── extraction/
│   │   ├── graph/
│   │   └── rag/
│   ├── models/
│   ├── config/
│   └── main.py
│
├── frontend/
│   ├── pages/
│   ├── components/
│   └── app.py
│
├── requirements.txt
├── .env
└── README.md
```

---

# Functional Requirements

# 1. Document Upload

Users can upload:

- `.pdf`
- `.docx`
- `.txt`

documents through the web interface.

## Upload Workflow

1. User uploads file from Streamlit frontend.
2. Frontend sends multipart request to:
   - `POST /documents/upload`
3. Backend returns:
   - `202 Accepted`
   - `document_id`
4. Processing begins asynchronously.
5. Frontend displays processing status.
6. Final status becomes:
   - `completed`
   - or `failed`

---

# 2. Document Parsing

The parser extracts raw UTF-8 text from uploaded documents.

## Supported Formats

| Format | Extraction Behavior |
|---|---|
| PDF | Extract all text |
| DOCX | Extract paragraphs and tables |
| TXT | Read raw UTF-8 text |

## Parsing Rules

- Preserve paragraph boundaries.
- Raise descriptive parsing exceptions on corrupted files.

---

# 3. Entity & Relationship Extraction

The Entity Extractor sends document text to the Ollama-hosted LLM with a structured extraction prompt.

## Extracted Entity Types

- Person
- Organization
- Contract
- Clause
- Date
- Jurisdiction
- Obligation

## Relationship Format

Each relationship includes:

```json
{
  "source_entity": "Party A",
  "relationship_type": "SIGNED",
  "target_entity": "Contract B"
}
```

## Reliability Rules

- LLM responses must be valid JSON.
- Retry extraction up to 2 additional times if parsing fails.
- Associate all extracted objects with the source `document_id`.

---

# 4. Knowledge Graph Storage

Entities and relationships are stored in Neo4j.

## Node Schema

```json
{
  "id": "uuid",
  "name": "Alice Corp",
  "type": "Organization",
  "document_id": "doc_123"
}
```

## Relationship Schema

```json
{
  "type": "SIGNED",
  "document_id": "doc_123"
}
```

## Graph Rules

- Merge duplicate entities using:
  - `name`
  - `type`
- Enforce uniqueness constraints.
- Replace old graph data when re-uploading documents with the same filename.

---

# 5. Chat Interface

Users can query uploaded legal documents using natural language.

## Chat Workflow

1. User submits query.
2. Frontend sends:
   - query
   - chat history
3. Backend retrieves relevant graph context.
4. Backend builds Graph-RAG prompt.
5. Ollama LLM generates grounded response.
6. Frontend displays updated conversation.

---

# 6. Graph-RAG Query Grounding

The LLM must answer only using graph-derived context.

## Retrieval Rules

- Retrieve graph nodes within 2 hops of matching entities.
- Format context as triples:

```text
(Entity A) -[RELATIONSHIP]-> (Entity B)
```

## Hallucination Prevention

The system prompt explicitly instructs the LLM to:

- Use only provided graph context.
- Avoid fabricating information.

## Missing Context Handling

If no graph data exists:

- Backend informs the LLM.
- LLM informs the user no relevant data was found.

---

# 7. Document Management

The frontend displays uploaded documents with:

- filename
- upload timestamp
- processing status

## API Endpoints

### Get Documents

```http
GET /documents
```

### Get Graph Summary

```http
GET /documents/{document_id}/graph-summary
```

### Example Response

```json
{
  "nodes": 45,
  "edges": 67
}
```

---

# 8. System Configuration & Health

## Environment Variables

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

BACKEND_URL=http://localhost:8000
```

---

# API Endpoints

## Health Check

```http
GET /health
```

### Response

```json
{
  "status": "ok"
}
```

---

## Upload Document

```http
POST /documents/upload
```

---

## List Documents

```http
GET /documents
```

---

## Graph Summary

```http
GET /documents/{document_id}/graph-summary
```

---

## Chat Query

```http
POST /chat/query
```

### Example Request

```json
{
  "query": "Who signed the employment agreement?",
  "history": []
}
```

### Example Response

```json
{
  "response": "Alice Corp signed the Employment Agreement.",
  "history": []
}
```

---

# Installation

# 1. Clone Repository

```bash
git clone https://github.com/your-org/scatterbrain.git
cd scatterbrain
```

---

# 2. Create Virtual Environment

```bash
python -m venv venv
```

## Windows

```bash
venv\Scripts\activate
```

## Linux / macOS

```bash
source venv/bin/activate
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Configure Environment Variables

Create a `.env` file:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

BACKEND_URL=http://localhost:8000
```

---

# 5. Start Neo4j

Ensure Neo4j is running locally.

---

# 6. Start Ollama

Example:

```bash
ollama run llama3
```

---

# 7. Start Backend

```bash
uvicorn backend.main:app --reload
```

---

# 8. Start Frontend

```bash
streamlit run frontend/app.py
```

---

# Error Handling

## Upload Errors

- Unsupported file types rejected by frontend.
- Corrupted files raise parsing exceptions.

## Extraction Errors

- Malformed LLM responses trigger retries.
- Failed extractions update document status to `failed`.

## Chat Errors

- Backend errors displayed without clearing chat history.

---

# Non-Functional Considerations

## Scalability

- Async backend processing.
- Graph-based retrieval efficiency.

## Reliability

- Retry mechanisms for extraction.
- Dependency health verification at startup.

## Security

- Local-only LLM processing.
- No external document transmission required.

---

# Future Enhancements

- Authentication & user accounts.
- Multi-user graph isolation.
- Streaming chat responses.
- Vector embeddings hybrid retrieval.
- Advanced legal ontology support.
- Graph visualization UI.
- Citation tracing for answers.
- Document versioning.

---

# License

MIT License

---

# Authors

Ozzey Padayachee