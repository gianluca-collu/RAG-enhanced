
from __future__ import annotations

import json
import logging
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import MetadataMode, NodeRelationship, RelatedNodeInfo, TextNode

from .chunk_record import chunk_to_record

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """Write chunk nodes through the vector index owned by RagClient."""

    def __init__(self, index: VectorStoreIndex, embed_model: BaseEmbedding) -> None:
        self.index = index
        self.embed_model = embed_model

    def index_jsonl(self, path: str | Path) -> int:
        """Index chunks from JSONL file.

        Args:
            path: Path to JSONL file (one chunk per line).

        Returns:
            Number of chunks indexed.
        """
        chunks = self._load_jsonl(path)
        logger.info("Loaded %d chunks from %s", len(chunks), path)
        chunks_by_document: dict[str, list[dict]] = {}
        for chunk in chunks:
            document_id = chunk.get("doc_id")
            if not document_id:
                raise ValueError("Every JSONL chunk must include a stable doc_id")
            chunks_by_document.setdefault(str(document_id), []).append(chunk)
        return sum(
            self.index_chunks(document_chunks, document_id=document_id)
            for document_id, document_chunks in chunks_by_document.items()
        )

    def index_result(self, result) -> int:
        """Index chunks from a ChunkingResult directly.

        Args:
            result: ChunkingResult from chunker.

        Returns:
            Number of chunks indexed.
        """
        chunks = [chunk_to_record(chunk, result.source) for chunk in result.chunks]
        logger.info("Indexing %d chunks from chunker result", len(chunks))
        return self.index_chunks(chunks, document_id=result.doc_id)

    def index_chunks(self, chunks: list[dict], document_id: str = "unknown") -> int:
        """Embed chunks and upload to vector store backend.

        Args:
            chunks: List of chunk dicts with content, metadata.
            document_id: Document identifier for grouping chunks.

        Returns:
            Number of chunks successfully indexed.
        """
        if not chunks:
            logger.warning("No chunks to index")
            return 0
        if not document_id or document_id == "unknown":
            raise ValueError("A stable document_id is required to index and replace document chunks")

        nodes = []
        for position, chunk in enumerate(chunks):
            text = chunk.get("contextualized_content") or chunk.get("content", "")
            if not text.strip():
                continue

            breadcrumb = chunk.get("breadcrumb") or []
            metadata = dict(chunk.get("metadata") or {})
            metadata.update({
                "doc_id": document_id,
                "section_id": chunk.get("section_id", ""),
                "breadcrumb": " > ".join(str(part) for part in breadcrumb),
                "content_type": chunk.get("content_type", "TEXT"),
                "source": chunk.get("source", ""),
                "source_version": (chunk.get("metadata") or {}).get("source_version", ""),
                "token_count": chunk.get("token_count", 0),
            })
            if chunk.get("page") is not None:
                metadata["page"] = chunk["page"]
            for key, value in list(metadata.items()):
                if isinstance(value, (dict, list, tuple)):
                    metadata[key] = json.dumps(value, ensure_ascii=False)

            nodes.append(TextNode(
                id_=chunk.get("chunk_id") or f"{document_id}:{position}",
                text=text,
                metadata=metadata,
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id=document_id),
                },
            ))

        if nodes:
            texts = [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes]
            embeddings = self.embed_model.get_text_embedding_batch(texts)
            if len(embeddings) != len(nodes):
                raise RuntimeError("Embedding count does not match the number of indexed chunks")
            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding
            self.index.delete_ref_doc(document_id, delete_from_docstore=True)
            self.index.insert_nodes(nodes)
        logger.info("Indexed %d chunks", len(nodes))
        return len(nodes)

    @staticmethod
    def _load_jsonl(path: str | Path) -> list[dict]:
        """Load JSONL file (one chunk dict per line).

        Args:
            path: Path to JSONL file.

        Returns:
            List of chunk dicts.
        """
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

