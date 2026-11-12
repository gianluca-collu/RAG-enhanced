__version__ = "1.0.0"
__description__ = "A lightweight Azure-backed RAG chatbot framework"

from rag.config import Config
from rag.rag_client import RagClient
from rag.query import ChatEngine

__all__ = [
    "Config",
    "RagClient",
    "ChatEngine",
    "__version__",
]
