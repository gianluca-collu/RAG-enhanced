from __future__ import annotations

from typing import Optional, Dict, List, Any
from pathlib import Path
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.settings import Settings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.vector_stores.azureaisearch import IndexManagement

from rag.config import Config
from rag.utils import Logger


class RagClient:
    """Core RAG client for managing LLM, embeddings, and vector store."""
    
    def __init__(self, config: Config):
        """Initialize RAG client with configuration."""
        self.config = config
        self.logger = Logger("RagClient")
        
        self._llm: Optional[AzureOpenAI] = None
        self._embed_model: Optional[AzureOpenAIEmbedding] = None
        self._vector_store: Optional[AzureAISearchVectorStore] = None
        self._index: Optional[VectorStoreIndex] = None
        self._search_client: Optional[SearchClient] = None
        self._search_index_client: Optional[SearchIndexClient] = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize all components."""
        self.logger.info("Initializing RAG client...")
        
        try:
            self.logger.info("Creating LLM...")
            self._llm = self._create_llm()
            self.logger.info("✅ LLM initialized")
            
            self.logger.info("Creating embeddings model...")
            self._embed_model = self._create_embeddings()
            self.logger.info("✅ Embeddings initialized")
            
            self.logger.info("Creating search client...")
            self._search_client = self._create_search_client()
            self.logger.info("✅ Search client initialized")
            
            self.logger.info("Creating vector store...")
            self._vector_store = self._create_vector_store()
            self.logger.info("✅ Vector store initialized")
            
            self.logger.info("Loading vector index...")
            self._index = self._load_index()
            self.logger.info("✅ Vector index loaded")
            
            self.logger.info("RAG client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG client: {str(e)}")
            self.logger.error(f"Please verify your Azure credentials in .env file")
            raise
    
    def _create_llm(self) -> AzureOpenAI:
        """Create Azure OpenAI LLM instance."""
        try:
            llm = AzureOpenAI(
                model=self.config.openai_model,
                deployment_name=self.config.openai_deployment_name,
                api_key=self.config.openai_key,
                azure_endpoint=self.config.openai_url,
                api_version=self.config.openai_api_version,
                temperature=self.config.chat_temperature,
            )
            self.logger.info("LLM initialized successfully")
            return llm
        except Exception as e:
            self.logger.error(f"Failed to create LLM: {str(e)}")
            raise
    
    def _create_embeddings(self) -> AzureOpenAIEmbedding:
        """Create embeddings model instance."""
        try:
            embed_model = AzureOpenAIEmbedding(
                model=self.config.embedding_model,
                deployment_name=self.config.embedding_deployment_name,
                api_key=self.config.openai_key,
                azure_endpoint=self.config.openai_url,
                api_version=self.config.embedding_api_version,
            )
            self.logger.info("Embeddings model initialized successfully")
            return embed_model
        except Exception as e:
            self.logger.error(f"Failed to create embeddings: {str(e)}")
            raise
    
    def _create_search_client(self) -> SearchClient:
        """Create Azure AI Search client."""
        try:
            search_client = SearchClient(
                endpoint=self.config.search_url,
                index_name=self.config.index_name,
                credential=AzureKeyCredential(self.config.search_key)
            )
            self.logger.info(f"Search client connected to index: {self.config.index_name}")
            return search_client
        except Exception as e:
            self.logger.error(f"Failed to create search client: {str(e)}")
            raise
    
    def _create_vector_store(self) -> AzureAISearchVectorStore:
        """Create the configured index if missing, then connect its vector store."""
        try:
            self._search_index_client = SearchIndexClient(
                endpoint=self.config.search_url,
                credential=AzureKeyCredential(self.config.search_key),
            )
            vector_store = AzureAISearchVectorStore(
                search_or_index_client=self._search_index_client,
                id_field_key="chunk_id",
                chunk_field_key="chunk",
                embedding_field_key="embedding",
                metadata_string_field_key="metadata",
                doc_id_field_key="doc_id",
                index_name=self.config.index_name,
                index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
                embedding_dimensionality=self.config.embedding_dimensions,
            )
            self.logger.info("Vector store initialized for index: %s", self.config.index_name)
            return vector_store
        except Exception as e:
            self.logger.error(f"Failed to create vector store: {str(e)}")
            raise
    
    def _load_index(self) -> VectorStoreIndex:
        """Load vector index from storage context."""
        try:
            storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
            Settings.llm = self._llm
            Settings.embed_model = self._embed_model
            
            index = VectorStoreIndex.from_vector_store(
                vector_store=self._vector_store,
                storage_context=storage_context,
            )
            self.logger.info("Vector index loaded successfully")
            return index
        except Exception as e:
            self.logger.error(f"Failed to load vector index: {str(e)}")
            raise
    
    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        similarity_top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pipeline: question -> validate -> retrieve -> context -> answer -> sources."""
        try:
            validated_query = self._validate_query(query_text)
            retrieval_top_k = similarity_top_k or top_k or self.config.search_top_k
            retrieved_nodes = self._retrieve_sources(validated_query, retrieval_top_k)
            sources = self._format_sources(retrieved_nodes)
            context = self._build_context(sources)
            answer = self._generate_answer(validated_query, context, system_prompt)

            return {
                "answer": answer,
                "sources": sources,
                "query": validated_query,
            }
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            raise

    @staticmethod
    def _validate_query(query_text: str) -> str:
        if not query_text or not query_text.strip():
            raise ValueError("Query cannot be empty")
        return query_text.strip()

    def _retrieve_sources(self, query_text: str, top_k: int) -> list:
        retriever = self._index.as_retriever(similarity_top_k=top_k)
        return retriever.retrieve(query_text)

    @staticmethod
    def _format_sources(source_nodes: list) -> list[Dict[str, Any]]:
        return [
            {
                "content": item.node.get_content(),
                "metadata": item.node.metadata,
                "score": item.score,
            }
            for item in source_nodes
        ]

    @staticmethod
    def _build_context(sources: list[Dict[str, Any]]) -> str:
        context = "\n\n".join(
            f"[Fonte {position}]\n{source['content']}"
            for position, source in enumerate(sources, start=1)
        )
        return context or "Nessuna fonte pertinente è stata trovata."

    def _generate_answer(
        self,
        query_text: str,
        context: str,
        system_prompt: Optional[str],
    ) -> str:
        from rag.chat.prompts import RAG_SYSTEM_PROMPT

        response = self._llm.chat([
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=system_prompt or RAG_SYSTEM_PROMPT,
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=f"Fonti recuperate:\n{context}\n\nDomanda:\n{query_text}",
            ),
        ])
        return response.message.content or ""

    def ingest_file(self, file_path: str | Path) -> Dict[str, Any]:
        """Pipeline: file -> validate -> chunk -> index -> ingestion result."""
        source = Path(file_path)
        if not source.is_file():
            raise FileNotFoundError(f"Document not found: {source}")
        if not self.config.document_intelligence_endpoint or not self.config.document_intelligence_key:
            raise ValueError(
                "Set DOCUMENT_INTELLIGENCE_ENDPOINT and DOCUMENT_INTELLIGENCE_KEY to ingest documents"
            )
        if self._index is None:
            raise RuntimeError("Vector index is not initialized")

        from rag.client.chunking import DocumentChunker, IndexingPipeline

        document_chunker = DocumentChunker(self.config)
        chunking_result = document_chunker.run(source)
        indexing_pipeline = IndexingPipeline(self._index)
        indexed_count = indexing_pipeline.index_result(chunking_result)
        result = {
            "doc_id": chunking_result.doc_id,
            "source": chunking_result.source,
            "chunk_count": len(chunking_result.chunks),
            "indexed_count": indexed_count,
        }
        self.logger.info("Ingested %s: %d chunks indexed", source.name, indexed_count)
        return result
    
    @property
    def llm(self) -> AzureOpenAI:
        """Get LLM instance."""
        return self._llm
    
    @property
    def embed_model(self) -> AzureOpenAIEmbedding:
        """Get embeddings model instance."""
        return self._embed_model
    
    @property
    def vector_store(self) -> AzureAISearchVectorStore:
        """Get vector store instance."""
        return self._vector_store
    
    @property
    def index(self) -> VectorStoreIndex:
        """Get vector index."""
        return self._index
    
    @property
    def search_client(self) -> SearchClient:
        """Get search client."""
        return self._search_client
