from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from rag.api import app, get_rag_client


def configure_search_clients(monkeypatch, index_client, search_client):
    config = SimpleNamespace(
        search_url="https://search.example.test",
        search_key="test-key",
        index_name="knowledge-base",
    )
    monkeypatch.setattr("rag.api.Config.from_env", lambda: config)
    monkeypatch.setattr("rag.api.SearchIndexClient", lambda **kwargs: index_client)
    monkeypatch.setattr("rag.api.SearchClient", lambda **kwargs: search_client)


def test_health_does_not_require_rag_services():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_uses_rag_client(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-token")
    rag_client = MagicMock()
    rag_client.query.return_value = {
        "query": "How does ingestion work?",
        "answer": "It chunks and indexes documents.",
        "sources": [],
    }
    app.dependency_overrides[get_rag_client] = lambda: rag_client
    client = TestClient(app)
    try:
        response = client.post(
            "/search",
            headers={"Authorization": "Bearer test-token"},
            json={"query": "  How does ingestion work?  ", "top_k": 3},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["answer"] == "It chunks and indexes documents."
    rag_client.query.assert_called_once_with("How does ingestion work?", top_k=3)


def test_search_requires_bearer_token(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-token")
    response = TestClient(app).post("/search", json={"query": "test"})

    assert response.status_code == 401


def test_ingest_upload_calls_rag_chunker(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-token")
    rag_client = MagicMock()
    rag_client.ingest_file.side_effect = lambda path: {
        "doc_id": "doc-123",
        "source": Path(path).name,
        "chunk_count": 4,
        "indexed_count": 4,
    }
    app.dependency_overrides[get_rag_client] = lambda: rag_client
    client = TestClient(app)
    try:
        response = client.post(
            "/ingest",
            headers={"Authorization": "Bearer test-token"},
            files={"file": ("manual.pdf", b"pdf test bytes", "application/pdf")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["indexed_count"] == 4
    rag_client.ingest_file.assert_called_once()


def test_ingest_rejects_unsupported_file_type(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-token")
    app.dependency_overrides[get_rag_client] = lambda: MagicMock()
    client = TestClient(app)
    try:
        response = client.post(
            "/ingest",
            headers={"Authorization": "Bearer test-token"},
            files={"file": ("notes.txt", b"text", "text/plain")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 415


def test_delete_index_requires_explicit_confirmation(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-token")
    index_client = MagicMock()
    search_client = MagicMock()
    configure_search_clients(monkeypatch, index_client, search_client)

    response = TestClient(app).delete(
        "/index",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 400
    index_client.delete_index.assert_not_called()


def test_delete_index_deletes_configured_index(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-token")
    index_client = MagicMock()
    search_client = MagicMock()
    configure_search_clients(monkeypatch, index_client, search_client)

    response = TestClient(app).delete(
        "/index?confirm=true",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 200
    assert response.json() == {"index": "knowledge-base", "deleted": True}
    index_client.delete_index.assert_called_once_with("knowledge-base")


def test_count_documents_counts_unique_document_ids(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-token")
    index_client = MagicMock()
    search_client = MagicMock()
    search_client.search.return_value = [
        {"doc_id": "doc-a"},
        {"doc_id": "doc-a"},
        {"doc_id": "doc-b"},
        {"doc_id": None},
    ]
    configure_search_clients(monkeypatch, index_client, search_client)

    response = TestClient(app).get(
        "/index/documents",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 200
    assert response.json() == {"index": "knowledge-base", "document_count": 2}
    search_client.search.assert_called_once_with(search_text="*", select=["doc_id"])