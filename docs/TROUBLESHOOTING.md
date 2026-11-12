# RAGSimple Troubleshooting Guide

Run the commands below from the project root unless stated otherwise.

## Installation and imports

If Python cannot import `rag`, install the package from the project root:

```bash
pip install -e .
```

For development, including test and lint tools:

```bash
pip install -e ".[dev]"
```

## Environment configuration

Create a private `.env` file in the project root using the variable table in the README. Do not commit this file. Configure at least:

- `OPENAI_URL` and `OPENAI_KEY`
- `SEARCH_URL`, `SEARCH_KEY`, and `INDEX_NAME`

URLs must use `https://`. If the app reports missing configuration, run:

```bash
python scripts/validate_setup.py
```

## Azure connection and index issues

Check that the Azure OpenAI and Azure AI Search endpoints are reachable, the credentials are valid, and `INDEX_NAME` matches an existing index. To validate application configuration and client initialization, run:

```bash
ragsimple test
```

Use `LOG_LEVEL=DEBUG` when investigating connection failures. Check the `logs/` directory for details.

## Missing dependencies

Reinstall the project dependencies from the root:

```bash
pip install -e .
```

Avoid installing individual versions manually; the supported dependency constraints are maintained in `pyproject.toml`.

## Slow or poor responses

- Check Azure quota and AI Search resource consumption.
- Reduce `search_top_k` for large indexes.
- Review retrieval settings in `rag/config.py` and the prompt in `rag/chat/prompts.json`.
- Test with a small query before investigating larger documents or batches.

## Tests and logs

Run the automated test suite with:

```bash
pytest tests/ -v
```

The application creates `logs/` automatically. On Windows PowerShell, inspect recent logs with `Get-ChildItem logs` and `Get-Content <log-file> -Tail 100`.