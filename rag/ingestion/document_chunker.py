from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from openai import AzureOpenAI

from .chunk_record import chunk_to_record
from .document_intelligence import create_document_intelligence_client
from .vision import create_vision_client
from .steps.chunk_assembler import Chunk, ChunkAssembler
from .steps.document_normalizer import DocumentNormalizer, NormalizedDocument
from .steps.metadata_extractor import MetadataExtractor
from .steps.section_assembler import SectionAssembler
from .steps.semantic_enricher import SemanticEnricher
from .steps.structure_analyzer import Section, StructureAnalyzer
from .steps.vision_describer import VisionDescriber
from .steps.visual_processor import VisualProcessor




logger = logging.getLogger(__name__)


class _StructuredOpenAIClient:
    def __init__(self, settings) -> None:
        self.client = AzureOpenAI(
            api_key=settings.openai_key,
            api_version=settings.openai_api_version,
            azure_endpoint=settings.openai_url,
        )
        self.deployment = settings.openai_deployment_name

    def chat_structured(self, prompt: str, response_format):
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": "Return only a valid JSON object."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        payload = response.choices[0].message.content or "{}"
        return response_format.model_validate_json(payload)

@dataclass
class ChunkingResult:
    """Risultato completo della pipeline di chunking."""
    doc_id: str
    source: str
    markdown: str
    sections: list[Section]
    chunks: list[Chunk]


class DocumentChunker:
    """Pipeline: file -> Markdown normalizzato -> sezioni arricchite -> chunk."""

    def __init__(self, settings) -> None:
        """Inizializza tutti i componenti della pipeline."""

        self.settings = settings
        if not settings.document_intelligence_endpoint or not settings.document_intelligence_key:
            raise ValueError(
                "Azure Document Intelligence endpoint and key are required for document ingestion"
            )
        self.document_intelligence_client = create_document_intelligence_client(settings)

        vision_enabled = settings.enable_vision
        vision_client = create_vision_client(settings) if vision_enabled else None
        llm_client = _StructuredOpenAIClient(settings) if settings.enable_semantic_metadata else None

        self.normalizer = DocumentNormalizer(
            document_intelligence_client=self.document_intelligence_client, settings = self.settings
        )

        self.analyzer = StructureAnalyzer()

        self.vision_describer = VisionDescriber(
            vision_client=vision_client,
            settings=settings,
        )

        self.visual_processor = (
            VisualProcessor(
                document_intelligence_client=self.document_intelligence_client,
                vision_describer=self.vision_describer,
                settings=settings,
            )
            if vision_enabled
            else None
        )

        self.assembler = ChunkAssembler(
            max_tokens=settings.chunk_size,
            overlap=settings.chunk_overlap,
            table_max_tokens=getattr(settings, "table_max_tokens", 800),
        )

        self.section_assembler = SectionAssembler()

        self.metadata_extractor = MetadataExtractor()

        self.semantic_enricher = SemanticEnricher(
            client=llm_client,
            settings=settings,
        )

    def export_jsonl(self, result: ChunkingResult, path: str | Path) -> None:
        """Write chunking result to JSONL file (one chunk per line)."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            for chunk in result.chunks:
                record = chunk_to_record(chunk, result.source)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("KB scritta in %s (%d chunk)", path, len(result.chunks))
    
    def run(self, source: str | Path) -> ChunkingResult:
        """Run the document-to-chunks pipeline."""
        start_time = time.time()
        source = Path(source)
        doc_id = self._file_hash(source)
        doc_type = source.suffix.lower().lstrip(".")

        logger.info("%s (.%s)", source.name, doc_type)
        # Step 1: normalize the source document.
        normalized_document = self.normalizer.normalize(source)
        logger.info(
            "   Pagine: %d | Markdown: %d car",
            len(getattr(normalized_document.result, "pages", [])),
            len(normalized_document.markdown),
        )

        # Step 2: analyze the normalized Markdown into sections.
        analyzed_sections = self.analyzer.analyze(normalized_document.markdown, doc_id)
        visual_sections = analyzed_sections

        # Step 3: enrich sections with visual descriptions when enabled.
        if self.visual_processor is not None:
            visual_sections = self.visual_processor.process(normalized_document, analyzed_sections)

        # Step 4: assemble the analyzed sections into a document structure.
        assembled_sections = self.section_assembler.assemble(visual_sections)

        # Step 5: add source metadata to the assembled sections.
        sections_with_metadata = self.metadata_extractor.extract(assembled_sections, source)

        # Step 6: enrich the sections with semantic metadata.
        enriched_sections = self.semantic_enricher.enrich(sections_with_metadata)

        # Step 7: create the final chunks.
        chunks = self.assembler.assemble(enriched_sections)

        if not chunks:
            total_blocks = sum(len(section.blocks) for section in enriched_sections)
            logger.warning("   0 chunk | %d sezioni | %d blocchi totali", len(chunks), total_blocks)

        elapsed = time.time() - start_time
        logger.info("   %d chunk | %.2fs", len(chunks), elapsed)

        return ChunkingResult(
            doc_id=doc_id,
            source=source.name,
            markdown=normalized_document.markdown,
            sections=enriched_sections,
            chunks=chunks,
        )

    def _file_hash(self, source: Path) -> str:
        """
        Calcola hash SHA1 del file.
        
        Legge il file a blocchi per evitare di caricarlo tutto in memoria.
        """
        digest = hashlib.sha1()

        with source.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

        return digest.hexdigest()

