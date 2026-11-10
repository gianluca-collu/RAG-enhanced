"""Tests for RAG Client."""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from llama_index.core.base.llms.types import MessageRole

from rag.chat.prompts import RAG_SYSTEM_PROMPT
from rag.client import RagClient


class TestRagClient:
    """Test suite for RagClient."""
    
    def test_initialization(self, mock_config):
        """Test client initialization."""
        components = [MagicMock() for _ in range(5)]
        with patch.object(RagClient, "_create_llm", return_value=components[0]), \
             patch.object(RagClient, "_create_embeddings", return_value=components[1]), \
             patch.object(RagClient, "_create_search_client", return_value=components[2]), \
             patch.object(RagClient, "_create_vector_store", return_value=components[3]), \
             patch.object(RagClient, "_load_index", return_value=components[4]):
            client = RagClient(mock_config)

        assert client.llm is components[0]
        assert client.embed_model is components[1]
        assert client.search_client is components[2]
        assert client.vector_store is components[3]
        assert client.index is components[4]
    
    def test_query_with_empty_string(self, mock_rag_client):
        """Test query with empty string."""
        client = RagClient.__new__(RagClient)
        client.logger = MagicMock()
        with pytest.raises(ValueError):
            client.query("")
    
    def test_query_success(self, mock_rag_client):
        """Test successful query."""
        result = mock_rag_client.query("test query")
        assert "answer" in result
        assert "sources" in result
        mock_rag_client.query.assert_called_once_with("test query")
    
    def test_properties(self, mock_rag_client):
        """Test property accessors."""
        mock_rag_client._llm = MagicMock()
        mock_rag_client.llm = mock_rag_client._llm
        assert mock_rag_client.llm is not None

    def test_query_sends_json_system_prompt_and_retrieved_context(self):
        client = RagClient.__new__(RagClient)
        client.config = SimpleNamespace(search_top_k=4)
        client.logger = MagicMock()
        client._index = MagicMock()
        client._llm = MagicMock()
        client._llm.chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content="Risposta dalle fonti")
        )

        node = SimpleNamespace(
            node=SimpleNamespace(
                get_content=lambda: "La policy prevede trenta giorni.",
                metadata={"source": "policy.pdf"},
            ),
            score=0.91,
        )
        client._index.as_retriever.return_value.retrieve.return_value = [node]
        steps = MagicMock()
        steps.attach_mock(client._index.as_retriever, "create_retriever")
        steps.attach_mock(client._index.as_retriever.return_value.retrieve, "retrieve")
        steps.attach_mock(client._llm.chat, "generate")

        result = client.query("Qual è il termine?")

        assert [call[0] for call in steps.mock_calls] == [
            "create_retriever",
            "retrieve",
            "generate",
        ]
        messages = client._llm.chat.call_args.args[0]
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[0].content == RAG_SYSTEM_PROMPT
        assert "La policy prevede trenta giorni." in messages[1].content
        assert "Qual è il termine?" in messages[1].content
        assert result["answer"] == "Risposta dalle fonti"
        assert result["sources"][0]["metadata"] == {"source": "policy.pdf"}
