# Scatterbrain ⚖️🧠🤖

**Scatterbrain** is an advanced legal document intelligence and Graph-RAG platform tailored for complex regulatory frameworks, manuals (such as the SARB Currency and Exchanges Manual), and contracts. 

It combines domain-aware hierarchical document parsing, automated knowledge graph extraction into **Neo4j**, vector indexing via **ChromaDB**, and multi-provider conversational querying with **Text2Cypher** and Graph-RAG retrieval.

---

## 🌟 Key Features

- **Multi-Format Document Ingestion**:
  - Upload and parse `.pdf` (via `pdfplumber`), `.docx` (via `python-docx`), and `.txt` documents.
  - Background asynchronous task processing with live status tracking.

- **Domain-Aware Legal Chunking**:
  - Hierarchical provision chunker designed to preserve legal hierarchy, clauses, definitions, and provision paths (e.g., `B.4(B)(iv)`).
  - Standard recursive text chunking for vector-based workflows.

- **Dual Retrieval & Indexing Modes**:
  - **GraphRAG (Knowledge Graph)**: Extracts entities and relationships into Neo4j. Uses `neo4j-graphrag` `Text2CypherRetriever` with automatic Cypher query sanitization and keyword fallback.
  - **Standard RAG (Vector Search)**: Indexes chunk embeddings into ChromaDB for traditional semantic vector retrieval.

- **Multi-Provider LLM Support**:
  - **Ollama (Local)**: CPU/GPU-configurable local inference (e.g., `mistral` for extraction/answers and `qwen3.5:0.8b` for fast Text2Cypher).
  - **DeepSeek**: Cloud API support (`deepseek-chat`) for high-accuracy reasoning and Cypher generation.
  - **OpenRouter**: Free-tier model rotation (Nemotron, Gemma, etc.) with automatic failover and rate-limit recovery.
  - **Anthropic & OpenAI**: Optional paid-tier API keys for Claude and OpenAI models.

- **Streamlit Interactive UI**:
  - **Landing Page**: Project overview and quick-start guide.
  - **Upload Hub**: File uploader, indexing mode selector (GraphRAG vs. Standard RAG), processing progress polling, and per-document entity/relationship graph metrics.
  - **Chat Interface**: Conversational query assistant with backend selection, retrieval mode toggle, chat history memory, and an inspectable Cypher query viewer.

---

## 🏗️ Architecture & Project Structure

```text
scatterbrain/
├── backend/                        # FastAPI backend application
│   ├── models/                     # Pydantic data schemas & request/response models
│   │   ├── chat.py                 # Chat request & response schemas
│   │   ├── document.py             # Document records and upload models
│   │   └── entities.py             # Extracted entity and relationship models
│   ├── routers/                    # FastAPI route definitions
│   │   ├── chat.py                 # /chat/query endpoint
│   │   ├── documents.py            # /documents CRUD & graph summary endpoints
│   │   └── health.py               # /health check endpoint
│   ├── services/                   # Business logic and external integrations
│   │   ├── chat_service.py         # Multi-backend chat orchestration & prompt grounding
│   │   ├── document_parser.py      # PDF, DOCX, and TXT parsing
│   │   ├── document_service.py     # Ingestion orchestration & concurrency control
│   │   ├── entity_extractor.py     # Entity and relation extraction logic
│   │   ├── external_llm_adapters.py# Adapter layer for paid LLMs
│   │   ├── external_llm_client.py  # DeepSeek & OpenRouter async clients with model rotation
│   │   ├── extraction_service.py   # Neo4j graph batch loader & extraction pipeline
│   │   ├── graph_query_service.py  # Text2Cypher retriever, sanitizer & fallback search
│   │   ├── graph_schema_service.py # Neo4j schema & constraints setup
│   │   ├── graph_service.py        # Neo4j client connection and graph metrics
│   │   ├── ollama_adapters.py      # LangChain / neo4j-graphrag Ollama LLM adapter
│   │   ├── ollama_client.py        # Async client for Ollama generation & health checks
│   │   ├── provision_chunker.py    # Legal document provision boundary parser
│   │   └── text_chunker.py         # Standard text chunker
│   ├── tests/                      # Unit, integration, and property-based tests (Hypothesis)
│   ├── config.py                   # Pydantic Settings (.env configuration)
│   ├── main.py                     # FastAPI application lifespan & entry point
│   └── requirements.txt            # Backend Python dependencies
├── frontend/                       # Streamlit web application
│   ├── pages/                      # Multi-page views
│   │   ├── 1_Upload.py             # Document upload, status polling & graph summary
│   │   └── 2_Chat.py               # Conversational Chat UI with backend & RAG selectors
│   ├── tests/                      # Frontend client tests
│   ├── api_client.py               # HTTP client communicating with FastAPI backend
│   ├── app.py                      # Main Streamlit landing page
│   └── requirements.txt            # Frontend Python dependencies
├── .env.example                    # Environment variable configuration template
├── pytest.ini                      # Pytest test markers and configuration
├── test_graphrag_smoketest.py      # End-to-end GraphRAG pipeline smoke test
└── README.md                       # Project documentation
```

---

## ⚙️ Configuration & Environment Variables

Copy `.env.example` to create your local `.env` file:

```bash
cp .env.example .env
```

Key configuration options:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `NEO4J_URI` | Neo4j Bolt connection URI | `neo4j://127.0.0.1:7687` |
| `NEO4J_USERNAME` | Neo4j database user | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j database password | *required* |
| `OLLAMA_BASE_URL` | Ollama service base URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default Ollama model for extraction & chat | `qwen3.5:0.8b` / `mistral` |
| `OLLAMA_TEXT2CYPHER_MODEL` | Local model for Text2Cypher query generation | `qwen3.5:0.8b` |
| `OLLAMA_NUM_GPU` | GPU layers offloaded (0 for CPU-only) | `0` |
| `OLLAMA_MAX_PARALLEL` | Concurrent Ollama requests during ingestion | `4` |
| `DEEPSEEK_CHAT_API_KEY` | DeepSeek API Key (for DeepSeek backend option) | *optional* |
| `DEEPSEEK_CHAT_MODEL` | Model name for DeepSeek | `deepseek-chat` |
| `OPENROUTER_API_KEY` | OpenRouter API Key (for OpenRouter backend option) | *optional* |
| `ANTHROPIC_API_KEY` | Anthropic Claude key (for legacy auto-select Cypher) | *optional* |
| `OPENAI_API_KEY` | OpenAI key (for legacy auto-select Cypher) | *optional* |
| `BACKEND_URL` | Backend URL used by the Streamlit frontend | `http://localhost:8000` |

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **Neo4j 5.x** running locally or via Docker
- **Ollama** running locally (if using local models):
  ```bash
  ollama pull qwen3.5:0.8b
  ollama pull mistral
  ollama serve
  ```

### 2. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
*API documentation is interactively available at [http://localhost:8000/docs](http://localhost:8000/docs).*

### 3. Frontend Setup
In a new terminal:
```bash
# Navigate to frontend directory
cd frontend

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```
*Access the UI at [http://localhost:8501](http://localhost:8501).*

---

## 🧪 Testing & Verification

Run tests from the root or backend directory:

```bash
# Run all backend unit and property tests
pytest backend/tests

# Run frontend tests
pytest frontend/tests

# Run GraphRAG end-to-end smoke test
python test_graphrag_smoketest.py
```

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check for backend and service connectivity |
| `POST` | `/documents/upload` | Upload a document (`pdf`, `docx`, `txt`) with index type (`graphrag` or `standard_rag`) |
| `GET` | `/documents` | List all tracked documents and processing status |
| `GET` | `/documents/{document_id}` | Get status and metadata for a specific document |
| `GET` | `/documents/{document_id}/graph-summary` | Retrieve node and edge count metrics from Neo4j |
| `POST` | `/chat/query` | Send natural language query with chat history, backend choice, and RAG mode |