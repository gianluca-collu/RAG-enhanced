from typing import Dict, List, Any, Optional
from rag.client import RagClient
from rag.chat.prompts import RAG_SYSTEM_PROMPT
from rag.chat.memory import ChatMemoryManager
from rag.utils import Logger


class ChatEngine:
    """Conversational RAG engine with memory management."""
    
    def __init__(
        self,
        rag_client: RagClient,
        system_prompt: Optional[str] = None,
        enable_memory: bool = True,
        max_memory_messages: int = 20,
    ):
        """
        Initialize chat engine.
        
        Args:
            rag_client: RAG client instance
            system_prompt: Custom system prompt (uses RAG_SYSTEM_PROMPT by default)
            enable_memory: Whether to maintain chat history
            max_memory_messages: Maximum messages to keep in history
        """
        self.rag_client = rag_client
        self.system_prompt = system_prompt or RAG_SYSTEM_PROMPT
        self.enable_memory = enable_memory
        self.logger = Logger("ChatEngine")
        
        if enable_memory:
            self.memory = ChatMemoryManager(max_messages=max_memory_messages)
        else:
            self.memory = None
    
    def send_message(
        self,
        message: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send a message and get a response.
        
        Args:
            message: User message
            top_k: Number of search results to retrieve
        
        Returns:
            Dictionary with response, sources, and metadata
        """
        try:
            if not message or not message.strip():
                raise ValueError("Message cannot be empty")
            
            # Add to memory
            if self.enable_memory:
                self.memory.add_message("user", message)
            
            # Prepare context
            context_message = message
            if self.enable_memory and self.memory.has_messages():
                # Build context from previous messages
                previous_messages = self.memory.get_last_n_messages(5)
                context_lines = [msg["content"] for msg in previous_messages[:-1]]
                if context_lines:
                    context_message = "\n".join(context_lines) + f"\n\nNuova domanda: {message}"
            
            # Query RAG
            result = self.rag_client.query(
                context_message,
                top_k=top_k,
                system_prompt=self.system_prompt,
            )
            
            # Add response to memory
            if self.enable_memory:
                self.memory.add_message("assistant", result["answer"])
            
            return {
                "answer": result["answer"],
                "sources": result["sources"],
                "query": message,
                "has_context": self.enable_memory and self.memory.has_messages(),
            }
        except Exception as e:
            self.logger.error(f"Error in send_message: {str(e)}")
            raise
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get complete chat history."""
        if self.enable_memory:
            return self.memory.get_messages()
        return []
    
    def get_last_n_messages(self, n: int) -> List[Dict[str, str]]:
        """Get last n messages from history."""
        if self.enable_memory:
            return self.memory.get_last_n_messages(n)
        return []
    
    def clear_history(self):
        """Clear chat history."""
        if self.enable_memory:
            self.memory.clear()
            self.logger.info("Chat history cleared")
    
    def get_context_string(self) -> str:
        """Get formatted context string from history."""
        if self.enable_memory:
            return self.memory.get_context_string()
        return ""
    
    def set_system_prompt(self, prompt: str):
        """Update system prompt."""
        self.system_prompt = prompt
        self.logger.info("System prompt updated")
