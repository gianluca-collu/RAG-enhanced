# RAGSimple - Conversational RAG Platform

A lightweight, production-ready Retrieval-Augmented Generation (RAG) chatbot framework designed for conversational AI applications. Built with Azure AI Search, LlamaIndex, and Azure OpenAI.

## Features

- 🤖 **Conversational RAG**: Multi-turn chat with persistent memory
- 🔍 **Hybrid Search**: Vector + keyword search with Azure AI Search
- 💾 **Memory Management**: Built-in chat history with context awareness
- 🎯 **Prompt Engineering**: Customizable system prompts for different use cases
- 📊 **Streaming Support**: Real-time response streaming for better UX
- 🐳 **Docker Ready**: Production-ready containerization
- 🧪 **Test Suite**: Comprehensive unit and integration tests
- 📚 **Developer Friendly**: Clear setup instructions and API usage in this README

## Quick Start

### Installation

```bash
# Clone and setup
git clone <repo-url>
cd ragsimple

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Linux/Mac

# Install the package and its dependencies
pip install -e .
```

### Configuration

Create `.env` file:

```bash
cp .env.example .env
# Edit .env with your credentials:
# - OPENAI_URL: Azure OpenAI endpoint
# - OPENAI_KEY: Azure OpenAI API key
# - SEARCH_URL: Azure AI Search endpoint
# - SEARCH_KEY: Azure AI Search API key
# - INDEX_NAME: Name of the AI Search index
```

### Run Chatbot (Streamlit UI)

```bash
streamlit run rag/web/app.py
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
├── rag/                       # Main package
│   ├── client/               # Core RAG client
│   │   ├── rag_client.py
│   │   ├── chunking/          # Document-to-index pipeline
│   │   ├── document_intelligence/
│   │   └── vision/
│   ├── chat/                 # Chat engine
│   │   ├── chat_engine.py
│   │   ├── prompts.json
│   │   ├── prompts.py
│   │   └── memory.py
│   ├── utils/                # Utilities
│   │   └── logger.py
│   ├── api.py                # FastAPI HTTP endpoints
│   ├── web/                  # Streamlit UI
│   │   ├── app.py
│   │   └── config.json
│   ├── config.py             # Configuration
│   └── main.py               # CLI entry point
├── docs/                      # Guides and troubleshooting
├── scripts/                   # Development utilities
├── tests/                     # Test suite
└── pyproject.toml             # Packaging, dependencies and tooling
```

## Architecture

### Components

1. **RagClient**: Core RAG engine managing LLM, embeddings, and vector store
2. **ChatEngine**: Conversational layer with memory management
3. **Retrieval**: Query processing and source collection in `RagClient`
4. **Web UI**: Streamlit-based chatbot interface
5. **CLI**: Command-line interface for batch operations

### Flow

```
User Query -> validate -> retrieve nodes -> build context -> LLM + system prompt -> answer + sources
```

## API Reference

### RagClient

```python
from rag.client import RagClient

client = RagClient(
    openai_url="https://...",
    openai_key="...",
    search_url="https://...",
    search_key="...",
    index_name="my-index"
)

response = client.query("What is AI?")
```

### ChatEngine

```python
from rag.chat import ChatEngine

chat = ChatEngine(rag_client=client)
response = chat.send_message("Hello!")
history = chat.get_history()
chat.clear_memory()
```

## Document Ingestion

Set `DOCUMENT_INTELLIGENCE_ENDPOINT` and `DOCUMENT_INTELLIGENCE_KEY` in `.env`, then index a supported document with:

```bash
ragsimple ingest --file path/to/document.pdf
```

The configured Azure AI Search index must be available to the RAG client. Vision descriptions and semantic metadata are optional; enable them with `ENABLE_VISION=true` and `ENABLE_SEMANTIC_METADATA=true`.

## HTTP API

Set `API_KEY` in `.env`, then start the service with:

```bash
uvicorn rag.api:app --host 0.0.0.0 --port 8000
```

`GET /health` is public. `POST /search` accepts `{"query": "...", "top_k": 5}`; `POST /ingest` accepts a multipart file upload; `GET /index/documents` counts distinct document IDs, not chunks; and `DELETE /index?confirm=true` removes the configured index. All endpoints except health require `Authorization: Bearer <API_KEY>`. The index name is required in configuration and is created automatically if it does not exist. Interactive API documentation is available at `/docs`.

The RAG system prompt is editable in `rag/chat/prompts.json` and is sent to the model on every query.

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
docker run -p 8501:8501 \
  -e OPENAI_URL=<url> \
  -e OPENAI_KEY=<key> \
  -e SEARCH_URL=<url> \
  -e SEARCH_KEY=<key> \
  -e INDEX_NAME=<index> \
  ragsimple:latest
```

For both web and API services, create `.env` from `.env.example` and run `docker compose up --build`. Streamlit is available on port `8501` and FastAPI on port `8000`.

## Configuration

All configuration is managed through environment variables (.env file):

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_URL` | Azure OpenAI endpoint | Yes |
| `OPENAI_KEY` | Azure OpenAI API key | Yes |
| `SEARCH_URL` | Azure AI Search endpoint | Yes |
| `SEARCH_KEY` | Azure AI Search API key | Yes |
| `INDEX_NAME` | AI Search index name | Yes |
| `OPENAI_MODEL` | Model deployment name | No (default: gpt-4o) |
| `EMBEDDING_MODEL` | Embedding model name | No (default: text-embedding-ada-002) |
| `LOG_LEVEL` | Logging level | No (default: INFO) |

## Performance Tips

1. **Index Optimization**: Ensure AI Search index is properly configured with vector fields
2. **Batch Processing**: Use batch_query for multiple questions
3. **Caching**: Enable in production for frequently asked questions
4. **Memory Management**: Periodically clear chat history for long conversations
5. **Hybrid Search**: Combine vector and keyword search for better relevance

## Troubleshooting

See [the troubleshooting guide](docs/TROUBLESHOOTING.md) for common setup and runtime issues.
