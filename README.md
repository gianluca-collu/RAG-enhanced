# RAGSimple - Conversational RAG Platform

A Retrieval-Augmented Generation (RAG) application built with Azure AI Search, LlamaIndex, and Azure OpenAI.

## Features

- 🤖 **Conversational RAG**: Multi-turn chat with session-scoped memory
- 🔍 **Hybrid Search**: Vector + keyword search with Azure AI Search
- 💾 **Memory Management**: Built-in chat history with context awareness
- 🎯 **Prompt Engineering**: Customizable system prompts for different use cases
- 🐳 **Docker Compose**: Local development setup for Streamlit and FastAPI
- 🧪 **Test Suite**: Comprehensive unit and integration tests
- 📚 **Developer Friendly**: Clear setup instructions and API usage in this README

## Quick Start

### Installation

```bash
# Clone and setup
git clone <repo-url>
cd ragsimple

# Create and activate a virtual environment (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
# Install the package and development tools
python -m pip install -e ".[dev]"
```

### Configuration

Create a private `.env` file in the project root using the variables listed in the Configuration section below. Keep this file local; it contains credentials and is excluded from Git.

### Run Chatbot (Streamlit UI)

```bash
streamlit run rag/web_app.py
```

### Run with CLI

```bash
# Simple chat
ragsimple chat

# Batch query
ragsimple query --file queries.txt

# Ingest a document
ragsimple ingest --file path/to/document.pdf
```

## Project Structure

```
project/
├── rag/
│   ├── api.py, config.py, index_store.py, rag_client.py
│   ├── ingestion/             # Document Intelligence, parsing, chunking, indexing
│   ├── query/                 # Retrieval, chat memory, prompts, diagnostics export
│   ├── web_app.py             # Streamlit UI
│   └── main.py                # CLI
├── docs/
├── scripts/
├── tests/
├── docker-compose.yml
└── pyproject.toml
```

## Architecture

### Components

1. **RagClient**: creates Azure OpenAI, embedding, and Search clients.
2. **Ingestion**: normalizes documents with Document Intelligence, creates chunks, then indexes them.
3. **QueryEngine**: retrieves relevant chunks and generates an answer with Azure OpenAI.
4. **ChatEngine**: adds session-local conversational context for the Streamlit UI.
5. **API and Streamlit**: expose search/ingestion endpoints and a local chat interface.

### Flow

```text
Document -> Document Intelligence -> sections/chunks -> embeddings -> Azure AI Search
Question -> vector/hybrid retrieval -> source context -> Azure OpenAI answer
```

## API Reference

### ChatEngine

```python
from rag.config import load_config
from rag.rag_client import RagClient
from rag.query import ChatEngine

client = RagClient(load_config())
chat = ChatEngine(rag_client=client)
response = chat.send_message("Hello!")
history = chat.get_chat_history()
chat.clear_history()
```

## Document Ingestion

Set `DOCUMENT_INTELLIGENCE_ENDPOINT` and `DOCUMENT_INTELLIGENCE_KEY` in `.env`, then index a supported document with:

```bash
ragsimple ingest --file path/to/document.pdf
```

The configured Azure AI Search index must be available to the RAG client. During ingestion, `ENABLE_VISION=false` (default) skips figure extraction, image downloads, and Vision descriptions while Document Intelligence still extracts document text and layout. Set `ENABLE_VISION=true` to process figures with Azure OpenAI Vision. Semantic metadata is separately optional and can be enabled with `ENABLE_SEMANTIC_METADATA=true`.

Each document has a stable key: the filename by default, or an explicit `document_key` when uploading through the API. Re-ingesting with the same key replaces its previous chunks; use distinct keys for different documents that share a filename. The content hash is stored as `source_version`.

**Existing-index migration:** versions indexed before stable document keys and top-level `doc_id` were added must be backed up and re-ingested once. The existing metadata-only IDs cannot be removed by the new per-document replacement path. Use the protected `DELETE /index?confirm=true` only if you intend to rebuild the whole index.

## HTTP API

Set a unique `API_KEY` in `.env`, then start the API locally with:

```bash
uvicorn rag.api:app --host 127.0.0.1 --port 8000
```

`GET /health` is public. All other endpoints require `Authorization: Bearer <API_KEY>`. `POST /search` accepts `{"query": "...", "top_k": 5}`; `POST /ingest` accepts a multipart `file` and an optional `document_key`; `GET /index/documents` counts distinct document IDs, not chunks; and `DELETE /index?confirm=true` removes the configured index. Keep this development API on localhost unless you add production-grade identity, authorization, rate limits, and network controls. The index name is required in configuration and is created automatically if it does not exist. Interactive API documentation is available at `/docs`.

For retrieval diagnostics, set `enabled` to `true` in `rag/query/retrieval_debug.json`. Each query appends one row per retrieved chunk to `output/retrieval_chunks.xlsx`, including the question, rank, document, section, page, similarity score, and chunk text. Set it back to `false` to stop exporting. The workbook contains query and document text, so keep it local and handle it as sensitive data.

The RAG system prompt is editable in `rag/query/prompts.json` and is sent to the model on every query.

## Development

### Install Dev Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/ -v
pytest tests/ --cov=rag  # With coverage
```

### Code Quality

```bash
# Format code
black rag/ tests/

# Lint
ruff check rag/

# Type checking
mypy rag/
```

## Docker

### Build

```bash
docker build -t ragsimple:latest .
```

### Run

```bash
docker run --rm --env-file .env -p 127.0.0.1:8501:8501 ragsimple:latest
```

For both web and API services, create the private `.env` file using the Configuration table and run `docker compose up --build`. Both services bind to localhost: Streamlit on port `8501` and FastAPI on port `8000`.

## Configuration

All configuration is managed through environment variables (.env file):

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_URL` | Azure OpenAI endpoint | Yes |
| `OPENAI_KEY` | Azure OpenAI API key | Yes |
| `SEARCH_URL` | Azure AI Search endpoint | Yes |
| `SEARCH_KEY` | Azure AI Search API key | Yes |
| `INDEX_NAME` | AI Search index name | Yes |
| `OPENAI_MODEL` | LlamaIndex-supported model alias for metadata/context sizing; Azure routing uses the deployment below | No (default: gpt-4o) |
| `OPENAI_DEPLOYMENT_NAME` | Actual Azure chat deployment name | Yes |
| `OPENAI_API_VERSION` | Azure OpenAI API version | Yes |
| `API_KEY` | Bearer token protecting non-health API routes | Required for API |
| `EMBEDDING_MODEL` | Embedding model name | No (default: text-embedding-ada-002) |
| `EMBEDDING_DEPLOYMENT_NAME` | Actual Azure embeddings deployment name | Yes |
| `EMBEDDING_DIMENSIONS` | Vector size; must match model and index | No (default: 1536) |
| `DOCUMENT_INTELLIGENCE_ENDPOINT` | Azure Document Intelligence endpoint | Required for ingestion |
| `DOCUMENT_INTELLIGENCE_KEY` | Azure Document Intelligence key | Required for ingestion |
| `ENABLE_VISION` | Extract and describe figures during ingestion | No (default: false) |
| `LOG_LEVEL` | Logging level | No (default: INFO) |

## Performance Tips

1. **Evaluation**: Maintain representative questions and expected source sections.
2. **Retrieval**: Compare vector and hybrid modes using Recall@k or MRR before tuning chunking.
3. **Cost**: Track embedding and chat usage separately during ingestion and queries.
4. **Memory**: Chat history is session-local and resets when the Streamlit session ends.

## Troubleshooting

See [the troubleshooting guide](docs/TROUBLESHOOTING.md) for common setup and runtime issues.
