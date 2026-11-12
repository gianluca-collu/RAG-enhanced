from unittest.mock import MagicMock, patch

from llama_index.vector_stores.azureaisearch import IndexManagement

from rag.index_store import IndexStore


def test_vector_store_creates_missing_index_with_chunk_schema(mock_config):
    client = IndexStore.__new__(IndexStore)
    client.config = mock_config
    client.logger = MagicMock()
    index_client = MagicMock()
    vector_store = MagicMock()

    with patch(
        "rag.index_store.SearchIndexClient",
        return_value=index_client,
    ) as create_index_client:
        with patch(
            "rag.index_store.AzureAISearchVectorStore",
            return_value=vector_store,
        ) as create_vector_store:
            result = client._create_vector_store()

    assert result is vector_store
    create_index_client.assert_called_once()
    create_vector_store.assert_called_once_with(
        search_or_index_client=index_client,
        id_field_key="chunk_id",
        chunk_field_key="chunk",
        embedding_field_key="embedding",
        metadata_string_field_key="metadata",
        doc_id_field_key="doc_id",
        index_name=mock_config.index_name,
        index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
        embedding_dimensionality=mock_config.embedding_dimensions,
    )
