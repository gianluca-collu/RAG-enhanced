#!/usr/bin/env python
"""Validate the local RAGSimple development setup."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def check_imports():
    """Check that critical third-party imports work."""
    print("Checking Python imports...")

    imports_to_test = {
        "pydantic": "Pydantic (config)",
        "dotenv": "python-dotenv (env loading)",
        "azure.core.credentials": "Azure SDK core",
        "azure.search.documents": "Azure Search",
        "llama_index": "LlamaIndex",
        "llama_index.llms.azure_openai": "LlamaIndex Azure OpenAI",
        "llama_index.embeddings.azure_openai": "LlamaIndex Embeddings",
        "streamlit": "Streamlit",
        "click": "Click (CLI)",
    }

    failed = []
    for module, description in imports_to_test.items():
        try:
            __import__(module)
            print(f"  OK {description}")
        except ImportError as error:
            print(f"  FAIL {description}: {str(error)[:80]}")
            failed.append(description)

    return not failed


def check_structure():
    """Check that the expected source and test directories exist."""
    print("\nChecking directory structure...")

    required_dirs = [
        "rag/client",
        "rag/client/chunking",
        "rag/client/chunking/extractors",
        "rag/client/document_intelligence",
        "rag/client/vision",
        "rag/chat",
        "rag/utils",
        "rag/web",
        "tests",
    ]

    failed = []
    for dir_path in required_dirs:
        if (PROJECT_ROOT / dir_path).exists():
            print(f"  OK {dir_path}/")
        else:
            print(f"  FAIL {dir_path}/ - MISSING")
            failed.append(dir_path)

    return not failed


def check_env():
    """Check that .env exists and contains the required settings."""
    print("\nChecking environment configuration...")

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        print("  .env file not found; create it from .env.example")
        return False

    env_content = env_path.read_text(encoding="utf-8")
    required_keys = ["OPENAI_URL", "OPENAI_KEY", "SEARCH_URL", "SEARCH_KEY", "INDEX_NAME"]
    missing = []

    for key in required_keys:
        lines = [line for line in env_content.splitlines() if line.startswith(f"{key}=")]
        if lines and lines[0].split("=", 1)[1].strip() and "your" not in lines[0].lower():
            print(f"  OK {key} configured")
        else:
            print(f"  FAIL {key} missing or has a placeholder value")
            missing.append(key)

    return not missing


def check_config():
    """Check that application configuration loads."""
    print("\nTesting configuration loading...")

    try:
        from rag.config import Config

        config = Config.from_env(str(PROJECT_ROOT / ".env"))
        print("  OK Configuration loaded successfully")
        print(f"     Model: {config.openai_model}")
        print(f"     Index: {config.index_name}")
        return True
    except Exception as error:
        print(f"  FAIL Configuration loading: {str(error)[:100]}")
        return False


def check_rag_client():
    """Check that the RAG client can be imported without contacting Azure."""
    print("\nTesting RAG client import...")

    try:
        from rag.rag_client import RagClient

        print(f"  OK {RagClient.__name__} can be imported")
        return True
    except Exception as error:
        print(f"  FAIL Cannot import RagClient: {str(error)[:100]}")
        return False


def main():
    """Run all setup checks and return a process status."""
    results = {
        "Imports": check_imports(),
        "Structure": check_structure(),
        "Environment": check_env(),
        "Configuration": check_config(),
        "RAG client": check_rag_client(),
    }

    print("\nValidation results")
    for name, result in results.items():
        print(f"  {'PASS' if result else 'FAIL'} - {name}")

    passed = sum(results.values())
    print(f"Result: {passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())