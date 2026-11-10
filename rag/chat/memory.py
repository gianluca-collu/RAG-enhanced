from typing import List, Optional, Dict, Any
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer


class ChatMemoryManager:
    """Manager for chat conversation memory."""
    
    def __init__(self, max_messages: int = 20, token_limit: int = 3000):
        """
        Initialize chat memory manager.
        
        Args:
            max_messages: Maximum number of messages to keep
            token_limit: Maximum number of tokens to keep in memory
        """
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")
        self.max_messages = max_messages
        self.token_limit = token_limit
        self.memory = ChatMemoryBuffer.from_defaults(
            token_limit=token_limit,
            chat_history=[],
        )
    
    def add_message(self, role: str, content: str):
        """Add a message to memory."""
        roles = {"user": MessageRole.USER, "assistant": MessageRole.ASSISTANT}
        if role not in roles:
            raise ValueError(f"Unsupported chat role: {role}")
        self.memory.put(ChatMessage(role=roles[role], content=content))

        messages = self.memory.get_all()
        if len(messages) > self.max_messages:
            messages = messages[-self.max_messages:]
            if messages and messages[0].role == MessageRole.ASSISTANT:
                messages = messages[1:]
            self.memory.set(messages)
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from memory."""
        return [
            {"role": message.role.value, "content": message.content or ""}
            for message in self.memory.get()
        ]
    
    def get_last_n_messages(self, n: int) -> List[Dict[str, str]]:
        """Get last n messages."""
        if n <= 0:
            return []
        messages = self.get_messages()
        return messages[-n:] if len(messages) > n else messages
    
    def clear(self):
        """Clear all messages from memory."""
        self.memory.reset()
    
    def get_context_string(self) -> str:
        """Get messages as formatted string for context."""
        messages = self.get_messages()
        context_lines = [f"{msg['role'].upper()}: {msg['content']}" for msg in messages]
        return "\n".join(context_lines)
    
    def has_messages(self) -> bool:
        """Check if memory has any messages."""
        return len(self.get_messages()) > 0
