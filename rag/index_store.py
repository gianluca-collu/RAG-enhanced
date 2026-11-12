from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore, IndexManagement

from rag.config import Config
from rag.logger import Logger


class IndexStore:
    """Shared Azure AI Search and vector-index resources."""

    def __init__(self, config: Config, llm: AzureOpenAI, embed_model: AzureOpenAIEmbedding):
        self.config = config
        self.logger = Logger("IndexStore")
        self._search_client = self._create_search_client()
        self._vector_store = self._create_vector_store()
        Settings.llm = llm
        Settings.embed_model = embed_model
        self._index = self._load_index()

    def _create_search_client(self) -> SearchClient:
        return SearchClient(
            endpoint=self.config.search_url,
            index_name=self.config.index_name,
            credential=AzureKeyCredential(self.config.search_key),
        )

    def _create_vector_store(self) -> AzureAISearchVectorStore:
        self._search_index_client = SearchIndexClient(
            endpoint=self.config.search_url,
            credential=AzureKeyCredential(self.config.search_key),
        )
        return AzureAISearchVectorStore(
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

    def _load_index(self) -> VectorStoreIndex:
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        return VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            storage_context=storage_context,
        )

    @property
    def search_client(self) -> SearchClient:
        return self._search_client

    @property
    def search_index_client(self) -> SearchIndexClient:
        return self._search_index_client

    @property
    def vector_store(self) -> AzureAISearchVectorStore:
        return self._vector_store

    @property
    def index(self) -> VectorStoreIndex:
        return self._index
