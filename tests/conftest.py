"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import MagicMock
from rag.config import Config


@pytest.fixture
def mock_config():
    """Provide mock configuration."""
    return Config(
        openai_url="https://test.openai.azure.com/",
        openai_key="test-key-1234567890123456",
        search_url="https://test.search.windows.net/",
        search_key="test-search-key-1234567890",
        index_name="test-index",
    )


@pytest.fixture
def mock_rag_client(mock_config):
    """Provide mock RAG client."""
    mock_client = MagicMock()
    mock_client.config = mock_config
    mock_client.query = MagicMock(return_value={
        "answer": "Test response",
        "sources": [{"content": "Test source", "score": 0.9}],
        "query": "Test query"
    })
    return mock_client


@pytest.fixture
def mock_chat_engine(mock_rag_client):
    """Provide mock chat engine."""
    mock_engine = MagicMock()
    mock_engine.send_message = MagicMock(return_value={
        "answer": "Test response",
        "sources": [],
        "query": "Test query",
        "has_context": False
    })
    mock_engine.get_chat_history = MagicMock(return_value=[])
    return mock_engine
