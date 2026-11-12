"""Tests for RAG Client."""

import pytest
from openpyxl import load_workbook
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from rag.query.prompts import RAG_SYSTEM_PROMPT
from rag.query.query_engine import QueryEngine
from rag.query import retrieval_export
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
        client.config = SimpleNamespace(search_top_k=4, hybrid_search=True, min_relevance_score=None)
        client.logger = MagicMock()
        client._index = MagicMock()
        client._llm = MagicMock()
        client._llm.chat.side_effect = [
            SimpleNamespace(message=SimpleNamespace(content="How many organizations were contacted?")),
            SimpleNamespace(message=SimpleNamespace(content="Sono state contattate 74 organizzazioni.")),
        ]

        italian_node = TextNode(
            id_="italian-chunk",
            text="La policy prevede trenta giorni.",
            metadata={"source": "policy.pdf"},
        )
        english_node = TextNode(
            id_="agentic-chunk",
            text="A total of 74 organisations were contacted.",
            metadata={"source": "agentic.pdf"},
        )
        client._index.as_retriever.return_value.retrieve.side_effect = [
            [SimpleNamespace(node=italian_node, score=0.91)],
            [
                SimpleNamespace(node=english_node, score=0.90),
                SimpleNamespace(node=italian_node, score=0.89),
            ],
        ]
        steps = MagicMock()
        steps.attach_mock(client._index.as_retriever, "create_retriever")
        steps.attach_mock(client._index.as_retriever.return_value.retrieve, "retrieve")
        steps.attach_mock(client._llm.chat, "generate")

        query = "Quante organizzazioni sono state contattate?"
        result = client.query(query)

        assert [call[0] for call in steps.mock_calls] == [
            "create_retriever",
            "generate",
            "retrieve",
            "retrieve",
            "generate",
        ]
        assert client._llm.chat.call_count == 2
        retrieval_calls = client._index.as_retriever.return_value.retrieve.call_args_list
        assert [call.args[0] for call in retrieval_calls] == [
            query,
            "How many organizations were contacted?",
        ]
        messages = client._llm.chat.call_args_list[-1].args[0]
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[0].content == RAG_SYSTEM_PROMPT
        assert "La policy prevede trenta giorni." in messages[1].content
        assert "A total of 74 organisations were contacted." in messages[1].content
        assert "[Fonte 1; documento: policy.pdf" in messages[1].content
        assert query in messages[1].content
        assert result["answer"] == "Sono state contattate 74 organizzazioni."
        assert len(result["sources"]) == 2
        assert {source["metadata"]["source"] for source in result["sources"]} == {
            "policy.pdf",
            "agentic.pdf",
        }
        client._index.as_retriever.assert_called_once_with(
            similarity_top_k=4,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        )

    def test_english_query_is_also_retrieved_in_italian(self):
        engine = QueryEngine(
            config=SimpleNamespace(hybrid_search=True, min_relevance_score=None),
            index=MagicMock(),
            llm=MagicMock(),
        )
        engine.llm.chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content="Quante organizzazioni sono state contattate?")
        )
        english_node = TextNode(id_="english", text="organizations were contacted", metadata={})
        italian_node = TextNode(id_="italian", text="organizzazioni contattate", metadata={})
        engine.index.as_retriever.return_value.retrieve.side_effect = [
            [SimpleNamespace(node=english_node, score=0.9)],
            [SimpleNamespace(node=italian_node, score=0.8)],
        ]

        results = engine._retrieve_sources("How many organizations were contacted?", 5)

        calls = engine.index.as_retriever.return_value.retrieve.call_args_list
        assert [call.args[0] for call in calls] == [
            "How many organizations were contacted?",
            "Quante organizzazioni sono state contattate?",
        ]
        assert {result.node.node_id for result in results} == {"english", "italian"}

    def test_query_retrieval_applies_configured_score_cutoff(self):
        engine = QueryEngine(
            config=SimpleNamespace(search_top_k=3, hybrid_search=True, min_relevance_score=0.03),
            index=MagicMock(),
            llm=MagicMock(),
        )
        high = SimpleNamespace(score=0.04)
        low = SimpleNamespace(score=0.02)
        engine.index.as_retriever.return_value.retrieve.return_value = [high, low]

        results = engine._retrieve_sources("query", 3)

        assert results == [high]
        engine.index.as_retriever.assert_called_once_with(
            similarity_top_k=3,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        )

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

        with patch("rag.query.query_engine.export_retrieval") as export:
            result = engine.run("  Domanda  ")

        assert [call[0] for call in steps.mock_calls] == step_names
        states = [getattr(engine, name).call_args.args[0] for name in step_names]
        assert all(state is states[0] for state in states)
        assert result == {"answer": "Risposta", "sources": [], "query": "Domanda"}
        export.assert_called_once_with("Domanda", [])

    def test_retrieval_export_respects_json_flag_and_writes_ranked_chunks(self, tmp_path, monkeypatch):
        config_file = tmp_path / "retrieval_debug.json"
        output_file = tmp_path / "retrieval_chunks.xlsx"
        monkeypatch.setattr(retrieval_export, "CONFIG_FILE", config_file)
        monkeypatch.setattr(retrieval_export, "OUTPUT_FILE", output_file)
        sources = [
            {
                "content": "Chunk uno",
                "metadata": {"source": "bando.pdf", "section_path": "Art. 1", "page": 2},
                "score": 0.91,
            },
            {
                "content": "Chunk due",
                "metadata": {"source": "bando.pdf", "section_path": "Art. 3", "page": 4},
                "score": 0.84,
            },
        ]

        config_file.write_text('{"enabled": false}', encoding="utf-8")
        retrieval_export.export_retrieval("Come partecipo?", sources)
        assert not output_file.exists()

        config_file.write_text('{"enabled": true}', encoding="utf-8")
        retrieval_export.export_retrieval("Come partecipo?", sources)
        rows = list(load_workbook(output_file, read_only=True).active.values)

        assert rows[0][:7] == (
            "question", "rank", "document", "section_path", "page", "similarity_score", "chunk_text"
        )
        assert rows[1][:7] == ("Come partecipo?", 1, "bando.pdf", "Art. 1", 2, 0.91, "Chunk uno")
        assert rows[2][:7] == ("Come partecipo?", 2, "bando.pdf", "Art. 3", 4, 0.84, "Chunk due")
