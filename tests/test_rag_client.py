"""Tests for RAG Client."""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from llama_index.core.base.llms.types import MessageRole

from rag.query.prompts import RAG_SYSTEM_PROMPT
from rag.query.query_engine import QueryEngine
from rag.rag_client import RagClient


class TestRagClient:
    """Test suite for RagClient."""
    
    def test_initialization(self, mock_config):
        """Test client initialization."""
        components = [MagicMock() for _ in range(5)]
        index_store = MagicMock(
            search_client=components[2],
            search_index_client=MagicMock(),
            vector_store=components[3],
            index=components[4],
        )
        with patch.object(RagClient, "_create_llm", return_value=components[0]), \
             patch.object(RagClient, "_create_embeddings", return_value=components[1]), \
             patch("rag.rag_client.IndexStore", return_value=index_store):
            client = RagClient(mock_config)

        assert client.llm is components[0]
        assert client.embed_model is components[1]
        assert client.search_client is components[2]
        assert client.vector_store is components[3]
        assert client.index is components[4]
    
    def test_query_with_empty_string(self, mock_rag_client):
        """Test query with empty string."""
        client = RagClient.__new__(RagClient)
        client.config = SimpleNamespace(search_top_k=4)
        client.logger = MagicMock()
        client._index = MagicMock()
        client._llm = MagicMock()
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

    def test_query_pipeline_passes_state_through_ordered_steps(self):
        engine = QueryEngine(
            config=SimpleNamespace(search_top_k=4),
            index=MagicMock(),
            llm=MagicMock(),
            logger=MagicMock(),
        )
        engine._retrieve_sources = MagicMock(return_value=[])
        engine._generate_answer = MagicMock(return_value="Risposta")

        steps = MagicMock()
        step_names = [
            "_step_1_validate",
            "_step_2_retrieve",
            "_step_3_format_sources",
            "_step_4_build_context",
            "_step_5_generate_answer",
            "_step_6_build_result",
        ]
        for name in step_names:
            step = MagicMock(wraps=getattr(engine, name))
            setattr(engine, name, step)
            steps.attach_mock(step, name)

        result = engine.run("  Domanda  ")

        assert [call[0] for call in steps.mock_calls] == step_names
        states = [getattr(engine, name).call_args.args[0] for name in step_names]
        assert all(state is states[0] for state in states)
        assert result == {"answer": "Risposta", "sources": [], "query": "Domanda"}
