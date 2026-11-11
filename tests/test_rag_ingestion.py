from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from azure.ai.documentintelligence.models import AnalyzeOutputOption

from rag.ingestion.document_chunker import DocumentChunker
from rag.ingestion.indexing_pipeline import IndexingPipeline
from rag.ingestion.pipeline import IngestionPipeline
from rag.ingestion.steps.document_normalizer import DocumentNormalizer
from rag.ingestion.steps.section_assembler import SectionAssembler
from rag.ingestion.steps.structure_analyzer import Block, Section, StructureAnalyzer


@pytest.mark.parametrize("enable_vision", [False, True])
def test_document_normalizer_requests_figures_only_when_vision_is_enabled(tmp_path, enable_vision):
    source = tmp_path / "manuale.pdf"
    source.write_bytes(b"input")
    client = MagicMock()
    poller = MagicMock()
    poller.details = {"operation_id": "operation-1"}
    client.begin_analyze_document.return_value = poller
    normalizer = DocumentNormalizer(client, SimpleNamespace(enable_vision=enable_vision))

    normalizer._analyze_document(source)

    call_options = client.begin_analyze_document.call_args.kwargs
    if enable_vision:
        assert call_options["output"] == [AnalyzeOutputOption.FIGURES]
    else:
        assert "output" not in call_options


def test_structure_analyzer_tracks_pages_inside_a_section():
    markdown = "# Bando\nIntroduzione\n<!-- PageBreak -->\n## Art. 1\nPrima parte\n<!-- PageBreak -->\nSeconda parte"

    sections = StructureAnalyzer().analyze(markdown, "doc-1")
    article = next(section for section in sections if section.breadcrumb[-1] == "Art. 1")

    assert article.page == 2
    assert [(block.content, block.page) for block in article.blocks] == [
        ("Prima parte", 2),
        ("Seconda parte", 3),
    ]


def test_section_assembler_does_not_merge_distinct_tables_by_column_count():
    section = Section(
        section_id="section-1",
        doc_id="doc-1",
        breadcrumb=["Bando", "Art. 1"],
        blocks=[
            Block("TABLE", "| Nome | Valore |\n| --- | --- |\n| A | 1 |", page=1),
            Block("TEXT", "Seconda tabella", page=1),
            Block("TABLE", "| Codice | Stato |\n| --- | --- |\n| B | Attivo |", page=1),
        ],
    )

    assembled = SectionAssembler().assemble([section])

    tables = [block for block in assembled[0].blocks if block.type == "TABLE"]
    assert len(tables) == 2


def test_document_id_is_stable_for_the_same_logical_key():
    document_id = DocumentChunker._document_id("  Bando_1_GIUR_EC.pdf  ")

    assert document_id == DocumentChunker._document_id("bando_1_giur_ec.pdf")
    assert len(document_id) == 64


@pytest.mark.parametrize("enable_vision", [False, True])
def test_document_chunker_activates_vision_only_when_enabled(enable_vision):
    settings = SimpleNamespace(
        document_intelligence_endpoint="https://example.test",
        document_intelligence_key="placeholder",
        enable_vision=enable_vision,
        enable_semantic_metadata=False,
        chunk_size=1000,
        chunk_overlap=200,
        table_max_tokens=800,
    )
    document_client = MagicMock()

    with patch(
        "rag.ingestion.document_chunker.create_document_intelligence_client",
        return_value=document_client,
    ), patch("rag.ingestion.document_chunker.create_vision_client") as create_vision_client:
        chunker = DocumentChunker(settings)

    assert create_vision_client.called is enable_vision
    assert (chunker.visual_processor is not None) is enable_vision


def test_index_result_inserts_contextualized_chunks():
    index = MagicMock()
    embed_model = MagicMock()
    embed_model.get_text_embedding_batch.return_value = [[0.1, 0.2]]
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
    indexed_count = IndexingPipeline(index, embed_model).index_result(result)

    assert indexed_count == 1
    index.delete_ref_doc.assert_called_once_with("doc-1", delete_from_docstore=True)
    index.insert_nodes.assert_called_once()
    node = index.insert_nodes.call_args.args[0][0]
    assert node.id_ == "chunk-1"
    assert node.ref_doc_id == "doc-1"
    assert node.text == chunk.contextualized_content
    assert node.metadata["doc_id"] == "doc-1"
    assert node.metadata["keywords"] == '["installazione"]'
    assert node.embedding == [0.1, 0.2]


def test_index_chunks_requires_stable_document_id():
    with pytest.raises(ValueError, match="stable document_id"):
        IndexingPipeline(MagicMock(), MagicMock()).index_chunks(
            [{"content": "A sufficiently useful chunk."}]
        )


def test_document_chunker_runs_stages_in_order(tmp_path):
    source = tmp_path / "manuale.pdf"
    source.write_bytes(b"input")

    chunker = DocumentChunker.__new__(DocumentChunker)
    chunker._file_hash = MagicMock(return_value="version-1")
    chunker._document_id = MagicMock(return_value="doc-1")
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
    metadata_sections = [SimpleNamespace(metadata={})]
    enriched_sections = metadata_sections
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
    pipeline = IngestionPipeline(
        config,
        index=MagicMock(),
        logger=MagicMock(),
        embed_model=MagicMock(),
    )

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
