"""Tests for Chat Engine."""

import pytest
from unittest.mock import MagicMock
from rag.query import ChatEngine


class TestChatEngine:
    """Test suite for ChatEngine."""
    
    def test_initialization(self, mock_rag_client):
        """Test chat engine initialization."""
        engine = ChatEngine(rag_client=mock_rag_client, enable_memory=True)
        assert engine.rag_client == mock_rag_client
        assert engine.enable_memory is True
        assert engine.memory is not None
    
    def test_send_message_empty(self, mock_rag_client):
        """Test sending empty message."""
        engine = ChatEngine(rag_client=mock_rag_client)
        with pytest.raises(ValueError):
            engine.send_message("")
    
    def test_send_message_success(self, mock_rag_client):
        """Test successful message send."""
        engine = ChatEngine(rag_client=mock_rag_client)
        mock_rag_client.query.return_value = {
            "answer": "Test answer",
            "sources": [],
            "query": "Test query"
        }
        
        result = engine.send_message("Test query")
        assert result["answer"] == "Test answer"
        assert "query" in result
        mock_rag_client.query.assert_called_once()
        assert mock_rag_client.query.call_args.kwargs["system_prompt"] == engine.system_prompt
    
    def test_memory_management(self, mock_rag_client):
        """Test chat memory management."""
        engine = ChatEngine(rag_client=mock_rag_client, enable_memory=True)
        
        # Clear history
        engine.clear_history()
        history = engine.get_chat_history()
        assert len(history) == 0

        def test_memory_uses_chat_message_format_and_respects_limit(self, mock_rag_client):
            engine = ChatEngine(rag_client=mock_rag_client, max_memory_messages=2)
            engine.memory.add_message("user", "Prima domanda")
            engine.memory.add_message("assistant", "Prima risposta")
            engine.memory.add_message("user", "Seconda domanda")

            history = engine.get_chat_history()

            assert len(history) <= 2
            assert history[0] == {"role": "user", "content": "Seconda domanda"}

        def test_memory_rejects_non_chat_roles(self, mock_rag_client):
            engine = ChatEngine(rag_client=mock_rag_client)

            with pytest.raises(ValueError, match="Unsupported chat role"):
                engine.memory.add_message("system", "Prompt")
    
    def test_get_last_n_messages(self, mock_rag_client):
        """Test getting last n messages."""
        engine = ChatEngine(rag_client=mock_rag_client, enable_memory=True)
        messages = engine.get_last_n_messages(5)
        assert isinstance(messages, list)
    
    def test_system_prompt_update(self, mock_rag_client):
        """Test updating system prompt."""
        engine = ChatEngine(rag_client=mock_rag_client)
        new_prompt = "New system prompt"
        engine.set_system_prompt(new_prompt)
        assert engine.system_prompt == new_prompt
