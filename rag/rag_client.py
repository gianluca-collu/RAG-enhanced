from __future__ import annotations

from pathlib import Path
from typing import Optional
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from rag.config import Config
from rag.index_store import IndexStore
from rag.ingestion.pipeline import IngestionPipeline
from rag.logger import Logger
from rag.query.query_engine import QueryEngine


class RagClient:
    """Core RAG client for managing LLM, embeddings, and vector store."""

    def __init__(self, config: Config):
        """Initialize RAG client with configuration."""
        self.config = config
        self.logger = Logger("RagClient")

        self._llm: Optional[AzureOpenAI] = None
        self._embed_model: Optional[AzureOpenAIEmbedding] = None
        self._index_store: Optional[IndexStore] = None
        self._search_client = None
        self._search_index_client = None
        self._vector_store = None
        self._index = None

        self._initialize()

    def _initialize(self):
        """Initialize all components."""
        self.logger.info("Initializing RAG client...")

        try:
            self.logger.info("Creating LLM...")
            self._llm = self._create_llm()
            self.logger.info("LLM initialized")

            self.logger.info("Creating embeddings model...")
            self._embed_model = self._create_embeddings()
            self.logger.info("Embeddings initialized")

            self.logger.info("Connecting to search index...")
            self._index_store = IndexStore(self.config, self._llm, self._embed_model)
            self._search_client = self._index_store.search_client
            self._search_index_client = self._index_store.search_index_client
            self._vector_store = self._index_store.vector_store
            self._index = self._index_store.index
            self.logger.info("Search index initialized")

            self.logger.info("RAG client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG client: {str(e)}")
            self.logger.error("Please verify your Azure credentials in .env file")
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

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        similarity_top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Run the query workflow against the configured index."""
        validated_query = QueryEngine._validate_query(query_text)
        return QueryEngine(self.config, self._index, self._llm, self.logger).query(
            validated_query,
            top_k=top_k,
            similarity_top_k=similarity_top_k,
            system_prompt=system_prompt,
        )

    def ingest_file(self, file_path: str | Path, document_key: Optional[str] = None) -> dict:
        """Run the ingestion workflow for a local document."""
        return IngestionPipeline(
            self.config,
            self.index,
            self.logger,
            embed_model=self._embed_model,
        ).run(
            file_path,
            document_key=document_key,
        )

    @property
    def llm(self) -> AzureOpenAI:
        """Get LLM instance."""
        return self._llm

    @property
    def embed_model(self) -> AzureOpenAIEmbedding:
        """Get embeddings model instance."""
        return self._embed_model

    @property
    def vector_store(self):
        """Get vector store instance."""
        return self._vector_store

    @property
    def index(self):
        """Get vector index."""
        return self._index

    @property
    def search_client(self):
        """Get search client."""
        return self._search_client
