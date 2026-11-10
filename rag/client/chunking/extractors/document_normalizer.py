from __future__ import annotations

import logging
import re
from markdownify import markdownify
from dataclasses import dataclass
from pathlib import Path
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeOutputOption,
    DocumentContentFormat,
)

logger = logging.getLogger(__name__)


@dataclass
class NormalizedDocument:
    """
    Contiene l'output dello Step 1.

    markdown: testo Markdown usato dallo StructureAnalyzer.
    result: risultato completo restituito da Document Intelligence.
    operation_id: id necessario per recuperare le figure rilevate.
    """
    markdown: str
    result: object
    operation_id: str


class DocumentNormalizer:
    """Normalizza il documento usando Document Intelligence."""

    def __init__(self, document_intelligence_client: DocumentIntelligenceClient, settings) -> None:
        """
        Inizializza il normalizzatore.

        Document Intelligence viene chiamato una sola volta.
        Il Markdown viene usato nello Step 2.
        Il result e l'operation_id vengono riusati nello Step 3.
        """
        self.document_intelligence = document_intelligence_client
        self.settings = settings
        
    def normalize(self, source: Path) -> NormalizedDocument:
        """
        Esegue la normalizzazione del documento.

        Step 1: analizza il documento con Document Intelligence.
        Step 2: recupera il Markdown prodotto da DI.
        Step 3: pulisce il Markdown.
        Step 4: aggiunge il titolo del documento come heading # se mancante.
        Step 5: restituisce un NormalizedDocument riutilizzabile.
        
        Document Intelligence v4.0 (2024-11-30) supporta nativamente:
        PDF, DOCX, XLSX, PPTX, HTML, immagini (JPG, PNG, BMP, TIFF, HEIF).
        Niente conversione necessaria per questi formati.
        """
        result, operation_id = self._analyze_document(source)
        markdown = result.content or ""
        markdown, self.dropped_tables = self._filter_tables(markdown, result)    
        markdown = self._strip_figures(markdown)        
        markdown = self._convert_tables(markdown)
        markdown = self._clean_markdown(markdown)
        
        # Assicura che il documento abbia un titolo come heading # 
        # Se il markdown non inizia con # (heading di livello 1),
        # aggiunge il nome del file come titolo
        if markdown and not markdown.startswith("#"):
            doc_title = source.stem  # Nome del file senza estensione
            markdown = f"# {doc_title}\n\n{markdown}"
       
        return NormalizedDocument(
            markdown=markdown,
            result=result,
            operation_id=operation_id,
        )

    def _strip_figures(self, markdown: str) -> str:
        """Rimuove l'OCR delle figure dal markdown: la semantica e' gia'
        nei blocchi IMAGE descritti da Vision nello Step 3."""
        return re.sub(r"<figure\b[^>]*>.*?</figure>", "", markdown,
                      flags=re.DOTALL | re.IGNORECASE)

    def _classify_table(self, table, result, figures, counts) -> str | None:
        """Ritorna il motivo di scarto o None se è una tabella vera."""
        region = table.bounding_regions[0]
        page = result.pages[region.page_number - 1]
        _, top, _, bottom = self._box_edges(region)

        if self.settings.exclude_tables_inside_figures and self._inside_figure(region, figures):
            return "figure"
        if self.settings.exclude_layout_grids and self._is_layout_grid(table):
            return "layout"
        repeated_edge = (
            counts[self._signature(table)] >= 3
            and len(table.cells) <= 12
            and (top < page.height * 0.15 or bottom > page.height * 0.85)
        )
        if self.settings.exclude_page_boilerplate and repeated_edge:
            return "boilerplate"
        return None


    def _filter_tables(self, markdown: str, result):
        """Filtra le tabelle. Ritorna (markdown, metadata_scartati)."""
        tables = result.tables or []
        figures = result.figures or []

        counts = {}
        for table in tables:
            key = self._signature(table)
            counts[key] = counts.get(key, 0) + 1

        spans, metadata = [], []
        for table in tables:
            if not table.bounding_regions:
                continue
            reason = self._classify_table(table, result, figures, counts)
            if reason is None:
                continue                      # tabella vera: resta

            content = " ".join(c.content.strip() for c in table.cells)
            page = table.bounding_regions[0].page_number

            if self.settings.boilerplate_action == "metadata":
                metadata.append({"type": reason, "page": page, "content": content})
            # rimuovo dal markdown solo se NON la tengo inline
            if self.settings.boilerplate_action in ("remove", "metadata"):
                spans.extend(table.spans or [])

        unique = {(s.offset, s.length) for s in spans}
        for offset, length in sorted(unique, reverse=True):
            markdown = markdown[:offset] + markdown[offset + length:]

        logger.debug(f"Tabelle DI: {len(tables)} | scartate: {len(unique)} | metadata: {len(metadata)}")
        return markdown, metadata
    
    def _box_edges(self, region):
        xs, ys = region.polygon[0::2], region.polygon[1::2]
        return min(xs), min(ys), max(xs), max(ys)

    def _signature(self, table):
        text = " ".join(c.content.lower().strip() for c in table.cells)
        return re.sub(r"\d+", "#", text)
    
    def _is_layout_grid(self, table) -> bool:
        """Scarta griglie di impaginazione: celle vuote o riempite da simboli."""
        cells = table.cells or []
        if not cells:
            return True
        filler = {"", "☐", "☒", "☑", "□", "■", "•", "-", "x", "X"}
        weak = sum(1 for c in cells if c.content.strip() in filler)
        return weak / len(cells) >= 0.5
    
    def _inside_figure(self, table_region, figures) -> bool:
        tx1, ty1, tx2, ty2 = self._box_edges(table_region)
        area = max((tx2 - tx1) * (ty2 - ty1), 0.001)
        for figure in figures:
            fr = figure.bounding_regions[0]
            if fr.page_number != table_region.page_number:
                continue
            fx1, fy1, fx2, fy2 = self._box_edges(fr)
            overlap = (max(0, min(tx2, fx2) - max(tx1, fx1))
                       * max(0, min(ty2, fy2) - max(ty1, fy1)))
            if overlap / area >= 0.60:   # era 0.80
             return True
        return False
    
    def _analyze_document(self, source: Path):
        """
        Analizza il documento originale con Document Intelligence.

        Usa il modello prebuilt-layout.
        Richiede il contenuto in formato Markdown.
        Richiede anche l'output FIGURES per lo Step 3.
        Restituisce result completo e operation_id.
        """
        with open(source, "rb") as file:
            poller = self.document_intelligence.begin_analyze_document(
                "prebuilt-layout",
                body=file,
                output_content_format=DocumentContentFormat.MARKDOWN,
                output=[AnalyzeOutputOption.FIGURES],
            )

        result = poller.result()
        operation_id = poller.details["operation_id"]

        return result, operation_id
    
    def _convert_tables(self, markdown: str) -> str:
        """Converte le tabelle HTML in Markdown."""
        pattern = r"<table\b.*?</table>"
        tables = re.findall(pattern, markdown, flags=re.IGNORECASE | re.DOTALL)
        for table in tables:
            markdown = markdown.replace(table, markdownify(table, table_infer_header=True).strip())
        return markdown 

    def _clean_markdown(self, markdown: str) -> str:
        """
        Pulisce il Markdown prodotto da Document Intelligence.

        Uniforma i fine riga.
        Rimuove spazi e tab a fine riga.
        Riduce newline eccessivi.
        Restituisce Markdown pronto per lo Step 2.
        """
        markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")
        markdown = re.sub(r"[ \t]+\n", "\n", markdown)
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)

        return markdown.strip()
