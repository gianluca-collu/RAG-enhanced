
from __future__ import annotations

import json
import logging
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode

from .chunk_record import chunk_to_record

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """Write chunk nodes through the vector index owned by RagClient."""

    def __init__(self, index: VectorStoreIndex) -> None:
        self.index = index

    def index_jsonl(self, path: str | Path) -> int:
        """Index chunks from JSONL file.

        Args:
            path: Path to JSONL file (one chunk per line).

        Returns:
            Number of chunks indexed.
        """
        chunks = self._load_jsonl(path)
        logger.info("Loaded %d chunks from %s", len(chunks), path)
        document_id = chunks[0].get("doc_id", "unknown") if chunks else "unknown"
        return self.index_chunks(chunks, document_id=document_id)

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
            ))

        if nodes:
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

