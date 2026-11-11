import json
import hmac
import os
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Security, UploadFile
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rag.rag_client import RagClient
from rag.config import Config

load_dotenv()

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {
    ".bmp",
    ".docx",
    ".heif",
    ".html",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".pptx",
    ".tif",
    ".tiff",
    ".xlsx",
}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query cannot be blank")
        return value


app = FastAPI(title="RAGSimple API")
bearer_scheme = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def get_rag_client() -> RagClient:
    return RagClient(Config.from_env())


def get_search_clients():
    config = Config.from_env()
    credential = AzureKeyCredential(config.search_key)
    return (
        config,
        SearchIndexClient(endpoint=config.search_url, credential=credential),
        SearchClient(
            endpoint=config.search_url,
            index_name=config.index_name,
            credential=credential,
        ),
    )


def require_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> None:
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        raise HTTPException(status_code=503, detail="API_KEY is not configured")
    if credentials is None or not hmac.compare_digest(credentials.credentials, expected_key):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search")
def search(
    payload: SearchRequest,
    _: None = Depends(require_api_key),
    rag_client: RagClient = Depends(get_rag_client),
) -> dict:
    return rag_client.query(payload.query, top_k=payload.top_k)


@app.post("/ingest")
def ingest(
    file: UploadFile = File(...),
    document_key: Optional[str] = Form(default=None),
    _: None = Depends(require_api_key),
    rag_client: RagClient = Depends(get_rag_client),
) -> dict:
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported document type")
    document_key = (document_key or filename).strip()
    if not document_key or len(document_key) > 256:
        raise HTTPException(status_code=422, detail="document_key must contain 1 to 256 characters")

    with TemporaryDirectory(prefix="ragsimple-ingest-") as temp_dir:
        destination = Path(temp_dir) / filename
        size = 0
        with destination.open("wb") as output:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File exceeds 25 MB limit")
                output.write(chunk)

        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        return rag_client.ingest_file(destination, document_key=document_key)


@app.delete("/index")
def delete_index(
    confirm: bool = Query(default=False),
    _: None = Depends(require_api_key),
) -> dict:
    if not confirm:
        raise HTTPException(status_code=400, detail="Pass confirm=true to delete the index")

    config, index_client, _ = get_search_clients()
    try:
        index_client.get_index(config.index_name)
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Index not found") from exc

    index_client.delete_index(config.index_name)
    get_rag_client.cache_clear()
    return {"index": config.index_name, "deleted": True}


@app.get("/index/documents")
def count_documents(_: None = Depends(require_api_key)) -> dict:
    config, index_client, search_client = get_search_clients()
    try:
        index_client.get_index(config.index_name)
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Index not found") from exc

    document_ids = set()
    for result in search_client.search(search_text="*", select=["doc_id", "metadata"]):
        document_id = result.get("doc_id")
        if not document_id:
            metadata = result.get("metadata")
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = None
            if isinstance(metadata, dict):
                document_id = metadata.get("doc_id")
        if document_id:
            document_ids.add(str(document_id))
    return {"index": config.index_name, "document_count": len(document_ids)}
