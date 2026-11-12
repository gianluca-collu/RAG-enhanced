from __future__ import annotations

#import hashlib
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from azure.ai.documentintelligence import DocumentIntelligenceClient



from .document_normalizer import NormalizedDocument
from .structure_analyzer import Block, Section
from .vision_describer import VisionDescriber



logger = logging.getLogger(__name__)

class VisualProcessor:
    """Arricchisce le Section con blocchi visuali."""

    def __init__(
        self,
        document_intelligence_client: Optional[DocumentIntelligenceClient],
        vision_describer: VisionDescriber,
        settings,
    ) -> None:
        """
        Inizializza il processore visuale.

        Usa Document Intelligence solo per scaricare i crop delle figure.
        Usa VisionDescriber per classificare e descrivere le immagini.
        Usa settings per le soglie di filtro.
        """
        self.document_intelligence = document_intelligence_client
        self.vision_describer = vision_describer
        self.settings = settings


    def _figure_to_block(self,normalized: NormalizedDocument,figure,)-> Optional[Block]:
        """Trasforma una figura DI in un blocco visuale."""

        figure_id = getattr(figure, "id", "senza-id")
        page = figure.bounding_regions[0].page_number
            #if getattr(figure, "bounding_regions", None)
            #else None)

        if not self._is_useful_figure(normalized.result, figure):
            logger.debug(f"{figure_id} | pagina {page}: scartata dal filtro dimensionale")
            return None

        image_bytes = self._download_image(
            result=normalized.result,
            operation_id=normalized.operation_id,
            figure=figure,
        )

        if not image_bytes:
            logger.warning(f"{figure_id}: download immagine vuoto")
            return None

        described = self.vision_describer.describe(
            image_name=f"figure_{figure_id}.png",
            image_bytes=image_bytes,
        )

        if not described:
            logger.warning(f"{figure_id}: Vision non ha restituito una descrizione")
            return None

        kind, description = described

        logger.debug(f"{figure_id} | pagina {page}: descritta come {kind}")

        return Block(
            type=kind,
            content=description,
            page=figure.bounding_regions[0].page_number,
        )

    def _is_useful_figure(self, result, figure) -> bool:
        """
        Decide se una figura DI è utile per il RAG.

        Scarta figure senza id o bounding region.
        Recupera pagina e area occupata.
        Tiene solo figure sopra la soglia minima configurata.
        Non usa filtri header/footer separati per restare semplice.
        """
        if not getattr(figure, "id", None):
            return False

        if not getattr(figure, "bounding_regions", None):
            return False

        region = figure.bounding_regions[0]
        page = self._get_page(result, region.page_number)

        if not page:
            return False

        box = self._polygon_to_box(region.polygon)
        return self._area_ratio(box, page) >= self._min_visual_area_ratio()

    def _download_image(self,result,operation_id: str,figure,) -> bytes:
        """
        Scarica il crop della figura da Document Intelligence.

        Se l'immagine è vuota o già vista, restituisce bytes vuoti.
        Altrimenti registra l'hash e restituisce i bytes dell'immagine.
        """
        response = self.document_intelligence.get_analyze_result_figure(
            model_id=result.model_id,
            result_id=operation_id,
            figure_id=figure.id,
        )

        image_bytes = response.read() if hasattr(response, "read") else b"".join(response)

        return image_bytes


    def _section_for_page(self,sections: list[Section],page: int | None,) -> Section | None:
        """Trova la sezione della pagina o usa l'ultima disponibile."""

        for section in sections:
            if section.page == page:
                return section

        return sections[-1] if sections else None

    def _get_page(self, result, page_number: int):
        """
        Recupera la pagina dal result DI.

        Cerca dentro result.pages.
        I numeri pagina sono 1-based.
        Serve per calcolare l'area relativa della figura.
        """
        for page in getattr(result, "pages", []) or []:
            if page.page_number == page_number:
                return page

        return None

    def _polygon_to_box(self, polygon: list[float]) -> dict:
        """
        Converte un polygon DI in bounding box.

        Estrae coordinate minime e massime.
        Restituisce x_min, y_min, x_max, y_max.
        Serve per calcolare l'area della figura.
        """
        xs = polygon[0::2]
        ys = polygon[1::2]

        return {
            "x_min": min(xs),
            "y_min": min(ys),
            "x_max": max(xs),
            "y_max": max(ys),
        }

    def _area_ratio(self, box: dict, page) -> float:
        """
        Calcola l'area relativa della figura nella pagina.

        Valori bassi indicano spesso loghi, icone o decorazioni.
        Il valore viene confrontato con una soglia minima.
        Restituisce un rapporto tra area figura e area pagina.
        """
        box_width = box["x_max"] - box["x_min"]
        box_height = box["y_max"] - box["y_min"]

        figure_area = box_width * box_height
        page_area = page.width * page.height

        return figure_area / page_area if page_area else 0


    def _min_visual_area_ratio(self) -> float:
        """Restituisce la soglia minima di area visuale."""
        return getattr(self.settings, "min_visual_area_ratio", 0.01)

    def process(self, normalized: NormalizedDocument, sections: list[Section]) -> list:
        """Processa le figure DI e le aggiunge alle sezioni."""

        figures = getattr(normalized.result, "figures", []) or []
        logger.debug("Figure rilevate: %d", len(figures))

        # Download + Vision in parallelo (la parte lenta). map mantiene l'ordine.
        workers = getattr(self.settings, "vision_max_workers", 8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            blocks = list(pool.map(
                lambda fig: self._figure_to_block(normalized, fig), figures
            ))

        # L'attach alle sezioni resta in serie: niente race su section.blocks.
        added = 0
        for block in blocks:
            if block is None:
                continue
            section = self._section_for_page(sections, block.page)
            if section:
                section.blocks.append(block)
                added += 1

        logger.debug("Visuali aggiunte alle sezioni: %d", added)
        return sections
