from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict

from rag.config import Config
from rag.ingestion.document_chunker import ChunkingResult, DocumentChunker
from rag.ingestion.indexing_pipeline import IndexingPipeline
from rag.logger import Logger


@dataclass
class _IngestionState:
    source: Path
    document_key: str | None = None
    chunking_result: ChunkingResult | None = None
    indexed_count: int | None = None
    response: Dict[str, Any] | None = None


class IngestionPipeline:
    """Validate, chunk, and index a document in one workflow."""

    def __init__(
        self,
        config: Config,
        index: Any,
        logger: Logger | None = None,
        embed_model: Any = None,
    ):
        self.config = config
        self.index = index
        self.logger = logger or Logger("IngestionPipeline")
        self.embed_model = embed_model

    def run(self, file_path: str | Path, document_key: str | None = None) -> Dict[str, Any]:
        state = _IngestionState(Path(file_path), document_key=document_key)

        # Step 1: validate the document and required services.
        state = self._step_1_validate(state)
        # Step 2: transform the document into chunks.
        state = self._step_2_chunk(state)
        # Step 3: write those chunks to the search index.
        state = self._step_3_index(state)
        # Step 4: shape the pipeline output.
        state = self._step_4_build_result(state)
        assert state.response is not None
        return state.response

    def _step_1_validate(self, state: _IngestionState) -> _IngestionState:
        source = state.source
        if not source.is_file():
            raise FileNotFoundError(f"Document not found: {source}")
        if not self.config.document_intelligence_endpoint or not self.config.document_intelligence_key:
            raise ValueError(
                "Set DOCUMENT_INTELLIGENCE_ENDPOINT and DOCUMENT_INTELLIGENCE_KEY to ingest documents"
            )
        if self.index is None:
            raise RuntimeError("Vector index is not initialized")
        return state

    def _step_2_chunk(self, state: _IngestionState) -> _IngestionState:
        chunker = DocumentChunker(self.config)
        if state.document_key is None:
            state.chunking_result = chunker.run(state.source)
        else:
            state.chunking_result = chunker.run(state.source, document_key=state.document_key)
        return state

    def _step_3_index(self, state: _IngestionState) -> _IngestionState:
        assert state.chunking_result is not None
        if self.embed_model is None:
            raise RuntimeError("An embedding model is required for indexing")
        state.indexed_count = IndexingPipeline(self.index, self.embed_model).index_result(state.chunking_result)
        return state

    def _step_4_build_result(self, state: _IngestionState) -> _IngestionState:
        assert state.chunking_result is not None
        assert state.indexed_count is not None
        state.response = {
            "doc_id": state.chunking_result.doc_id,
            "source": state.chunking_result.source,
            "chunk_count": len(state.chunking_result.chunks),
            "indexed_count": state.indexed_count,
        }
        self.logger.info(
            f"Ingested {state.source.name}: {state.indexed_count} chunks indexed"
        )
        return state
