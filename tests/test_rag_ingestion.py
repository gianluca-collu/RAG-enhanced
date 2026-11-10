from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag.ingestion.document_chunker import DocumentChunker
from rag.ingestion.indexing_pipeline import IndexingPipeline
from rag.ingestion.pipeline import IngestionPipeline


def test_index_result_inserts_contextualized_chunks():
    index = MagicMock()
    chunk = SimpleNamespace(
        chunk_id="chunk-1",
        doc_id="doc-1",
        section_id="section-1",
        breadcrumb=["Manuale", "Installazione"],
        content="Installa il client.",
        contextualized_content="Documento: Manuale. Sezione: Installazione.\n\nInstalla il client.",
        content_type="TEXT",
        page=2,
        token_count=12,
        metadata={"keywords": ["installazione"]},
    )
    result = SimpleNamespace(doc_id="doc-1", source="manuale.pdf", chunks=[chunk])

    indexed_count = IndexingPipeline(index).index_result(result)

    assert indexed_count == 1
    index.insert_nodes.assert_called_once()
    node = index.insert_nodes.call_args.args[0][0]
    assert node.id_ == "chunk-1"
    assert node.text == chunk.contextualized_content
    assert node.metadata["doc_id"] == "doc-1"
    assert node.metadata["keywords"] == '["installazione"]'


def test_document_chunker_runs_stages_in_order(tmp_path):
    source = tmp_path / "manuale.pdf"
    source.write_bytes(b"input")

    chunker = DocumentChunker.__new__(DocumentChunker)
    chunker._file_hash = MagicMock(return_value="doc-1")
    chunker.normalizer = MagicMock()
    chunker.analyzer = MagicMock()
    chunker.visual_processor = None
    chunker.section_assembler = MagicMock()
    chunker.metadata_extractor = MagicMock()
    chunker.semantic_enricher = MagicMock()
    chunker.assembler = MagicMock()

    normalized = SimpleNamespace(
        markdown="# Manuale",
        result=SimpleNamespace(pages=[]),
    )
    raw_sections = ["raw"]
    structured_sections = ["structured"]
    metadata_sections = ["metadata"]
    enriched_sections = ["enriched"]
    chunks = ["chunk"]
    chunker.normalizer.normalize.return_value = normalized
    chunker.analyzer.analyze.return_value = raw_sections
    chunker.section_assembler.assemble.return_value = structured_sections
    chunker.metadata_extractor.extract.return_value = metadata_sections
    chunker.semantic_enricher.enrich.return_value = enriched_sections
    chunker.assembler.assemble.return_value = chunks

    stages = MagicMock()
    stages.attach_mock(chunker.normalizer.normalize, "normalize")
    stages.attach_mock(chunker.analyzer.analyze, "analyze")
    stages.attach_mock(chunker.section_assembler.assemble, "assemble_sections")
    stages.attach_mock(chunker.metadata_extractor.extract, "extract_metadata")
    stages.attach_mock(chunker.semantic_enricher.enrich, "enrich")
    stages.attach_mock(chunker.assembler.assemble, "make_chunks")

    result = chunker.run(source)

    assert [call[0] for call in stages.mock_calls] == [
        "normalize",
        "analyze",
        "assemble_sections",
        "extract_metadata",
        "enrich",
        "make_chunks",
    ]
    assert chunker.analyzer.analyze.call_args.args == (normalized.markdown, "doc-1")
    assert chunker.section_assembler.assemble.call_args.args == (raw_sections,)
    assert chunker.metadata_extractor.extract.call_args.args == (structured_sections, source)
    assert chunker.semantic_enricher.enrich.call_args.args == (metadata_sections,)
    assert chunker.assembler.assemble.call_args.args == (enriched_sections,)
    assert result.chunks == chunks


def test_ingestion_pipeline_passes_state_through_ordered_steps(tmp_path):
    source = tmp_path / "manuale.pdf"
    source.write_bytes(b"input")
    config = SimpleNamespace(
        document_intelligence_endpoint="https://example.test",
        document_intelligence_key="placeholder",
    )
    chunking_result = SimpleNamespace(
        doc_id="doc-1",
        source=source.name,
        chunks=["chunk-1", "chunk-2"],
    )
    chunker = MagicMock()
    chunker.run.return_value = chunking_result
    indexer = MagicMock()
    indexer.index_result.return_value = 2
    pipeline = IngestionPipeline(config, index=MagicMock(), logger=MagicMock())

    steps = MagicMock()
    step_names = [
        "_step_1_validate",
        "_step_2_chunk",
        "_step_3_index",
        "_step_4_build_result",
    ]
    for name in step_names:
        step = MagicMock(wraps=getattr(pipeline, name))
        setattr(pipeline, name, step)
        steps.attach_mock(step, name)

    with patch("rag.ingestion.pipeline.DocumentChunker", return_value=chunker), patch(
        "rag.ingestion.pipeline.IndexingPipeline", return_value=indexer
    ):
        result = pipeline.run(source)

    assert [call[0] for call in steps.mock_calls] == step_names
    states = [getattr(pipeline, name).call_args.args[0] for name in step_names]
    assert all(state is states[0] for state in states)
    chunker.run.assert_called_once_with(source)
    indexer.index_result.assert_called_once_with(chunking_result)
    assert result == {
        "doc_id": "doc-1",
        "source": "manuale.pdf",
        "chunk_count": 2,
        "indexed_count": 2,
    }