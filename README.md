# RAGSimple - Conversational RAG Platform

A Retrieval-Augmented Generation (RAG) application built with Azure AI Search, LlamaIndex, and Azure OpenAI.

## Features

- 🤖 **Conversational RAG**: Multi-turn chat with session-scoped memory
- 🔍 **Hybrid Search**: Vector + keyword search with Azure AI Search
- 🌐 **Multilingual Retrieval**: Italian-English query expansion with reciprocal-rank fusion
- 💾 **Memory Management**: Built-in chat history with context awareness
- 🎯 **Prompt Engineering**: Customizable system prompts for different use cases
- 🐳 **Docker Compose**: Local development setup for Streamlit and FastAPI
- 🧪 **Test Suite**: Automated API, retrieval, ingestion, and chat tests
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

Copy `.env.example` to `.env` in the project root, then fill in your Azure resource values and credentials. Keep `.env` local; it contains secrets and is excluded from Git.

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
3. **QueryEngine**: retrieves with the original and translated query, fuses ranked chunks, then generates an answer with Azure OpenAI.
4. **ChatEngine**: adds session-local conversational context for the Streamlit UI.
5. **API and Streamlit**: expose search/ingestion endpoints and a local chat interface.

### Flow

```text
Document -> Document Intelligence -> sections/chunks -> embeddings -> Azure AI Search
Question -> vector/hybrid retrieval -> source context -> Azure OpenAI answer
```

## Detailed guides

- [Deployment and access management](docs/DEPLOYMENT.md): Azure Container Apps, OIDC login, and an allowlist for inviting and revoking users.
- [Chunking](docs/CHUNKING.md): document extraction, sections, block types, and chunk construction rules.
- [Vectorization and search](docs/VECTORIZATION.md): embedded content, Azure AI Search, and multilingual retrieval.
- [Application architecture](docs/ARCHITECTURE.md): components and end-to-end document and question flows.

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

The local PDF corpus under `docs/KB/` is intentionally excluded from Git and Docker. Only redistribute source documents when their licenses permit it.

Set `DOCUMENT_INTELLIGENCE_ENDPOINT` and `DOCUMENT_INTELLIGENCE_KEY` in `.env`, then index a supported document with:

```bash
ragsimple ingest --file path/to/document.pdf
```

The configured Azure AI Search index must be available to the RAG client. During ingestion, `ENABLE_VISION=false` (default) skips figure extraction, image downloads, and Vision descriptions while Document Intelligence still extracts document text and layout. Set `ENABLE_VISION=true` to process figures with Azure OpenAI Vision. Semantic metadata is separately optional and can be enabled with `ENABLE_SEMANTIC_METADATA=true`.

Each document has a stable key: the filename by default, or an explicit `document_key` when uploading through the API. Re-ingesting with the same key replaces its previous chunks; use distinct keys for different documents that share a filename. The content hash is stored as `source_version`.

**Existing-index migration:** versions indexed before stable document keys and top-level `doc_id` were added must be backed up and re-ingested once. The existing metadata-only IDs cannot be removed by the new per-document replacement path. Use the protected `DELETE /index?confirm=true` only if you intend to rebuild the whole index.

## HTTP API

Set a unique `API_KEY` in `.env`, then start the API locally with:

Generate a local token with PowerShell:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Store that output as `API_KEY` in the private `.env`; do not reuse an Azure key.

```bash
uvicorn rag.api:app --host 127.0.0.1 --port 8000
```

`GET /health` is public. All other endpoints require `Authorization: Bearer <API_KEY>`. `POST /search` accepts `{"query": "...", "top_k": 5}` with `top_k` from 1 to 50; `POST /ingest` accepts a multipart `file` and an optional `document_key`; `GET /index/documents` counts distinct document IDs, not chunks; and `DELETE /index?confirm=true` removes the configured index. Keep this development API on localhost unless you add production-grade identity, authorization, rate limits, and network controls. The index name is required in configuration and is created automatically if it does not exist. Interactive API documentation is available at `/docs`.

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
### Online access and account management

The container image requires authentication by default. Local Docker Compose explicitly defaults `AUTH_REQUIRED` to `false` because its ports are bound to localhost. For an online deployment, set `AUTH_REQUIRED=true`, configure an OpenID Connect (OIDC) provider, and expose only the Streamlit service on HTTPS. The API should remain private unless a separate authenticated use case requires it.

Configure these values as environment variables in the hosting platform; store `OIDC_COOKIE_SECRET` and `OIDC_CLIENT_SECRET` as platform secrets:

- `OIDC_REDIRECT_URI`: the public app URL ending in `/oauth2callback`.
- `OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET`, and `OIDC_METADATA_URL`: values from the chosen OIDC provider (for example, Microsoft Entra ID or Google).
- `OIDC_COOKIE_SECRET`: a strong random secret used to sign the login cookie.
- `ALLOWED_USER_EMAILS`: comma-separated email addresses allowed to use the app.

The app writes Streamlit's OIDC `secrets.toml` at container startup from these runtime values; it is not included in the Docker image or Git. To grant access, add an email to `ALLOWED_USER_EMAILS` in the hosting platform and activate the updated configuration. To revoke access, remove it and activate the update; the allowlist is checked on each app interaction, so an existing login cookie does not bypass removal. Do not enable public self-registration without usage controls: model and search costs are charged to the application's Azure resources.

For Azure Container Apps, deploy the web container with external HTTPS ingress on port `8501`, configure its OIDC values and allowlist, and leave the API without public ingress. Set the OIDC provider's callback URL to the exact `OIDC_REDIRECT_URI` value.

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
| `AUTH_REQUIRED` | Require OIDC login for Streamlit | No (Docker image defaults to `true`; local Compose defaults to `false`) |
| `ALLOWED_USER_EMAILS` | Comma-separated allowlist of Streamlit user emails | Required when Streamlit authentication is enabled |
| `OIDC_REDIRECT_URI` | Absolute OIDC callback URL ending in `/oauth2callback` | Required when Streamlit authentication is enabled |
| `OIDC_COOKIE_SECRET` | Random secret for Streamlit's authentication cookie | Required when Streamlit authentication is enabled |
| `OIDC_CLIENT_ID` | OIDC application client ID | Required when Streamlit authentication is enabled |
| `OIDC_CLIENT_SECRET` | OIDC application client secret | Required when Streamlit authentication is enabled |
| `OIDC_METADATA_URL` | OIDC discovery document URL | Required when Streamlit authentication is enabled |
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

## Retrieval Evaluation

Create a local JSON dataset with the expected section paths from the indexed documents:

```json
[
	{
		"question": "Come si presenta la domanda?",
		"relevant_sections": ["Bando > Art. 3 Presentazione della domanda"]
	}
]
```

Run `python -m rag.evaluation eval/retrieval.json --top-k 5`. The tool evaluates retrieval only (no chat completion) and reports Recall@5 and MRR@5. It calls Azure embeddings for each question. Keep the question dataset local; `eval/retrieval.json` is ignored by Git.

## Troubleshooting

See [the troubleshooting guide](docs/TROUBLESHOOTING.md) for common setup and runtime issues.
