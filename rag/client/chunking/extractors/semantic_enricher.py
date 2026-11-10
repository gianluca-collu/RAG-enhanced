import json
import logging
from typing import Any, List, Dict, Optional
from pydantic import BaseModel, Field

from .structure_analyzer import Section

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Sei un assistente che indicizza documenti aziendali per una knowledge base.\n"
    "Dato il testo di una sezione, rispondi SOLO con un oggetto JSON valido."
)

USER_PROMPT_TEMPLATE = (
    "Analizza questo testo e fornisci:\n"
    '- "summary": stringa di 1-2 frasi in italiano che riassume il contenuto;\n'
    '- "keywords": lista di 3-8 parole o brevi frasi chiave in italiano;\n'
    '- "topics": lista di 1-5 argomenti generali trattati;\n'
    '- "entities": oggetto con liste "organizations", "people", "locations",\n'
    '              "dates", "money" (usa liste vuote se non ci sono);\n'
    '- "cited_documents": lista di nomi di documenti, allegati o riferimenti\n'
    '                     citati nel testo (es: "Documento X", "Allegato A",\n'
    '                     "Report Y", [DOC-123]). Usa lista vuota se nessuno.\n'
    "\nTesto da analizzare:\n{text}"
)


# Pydantic model for structured output
class SemanticMetadata(BaseModel):
    """Metadati semantici estratti dall'LLM."""
    summary: str = Field(description="Riassunto di 1-2 frasi")
    keywords: List[str] = Field(description="3-8 parole chiave")
    topics: List[str] = Field(description="1-5 argomenti generali")
    organizations: List[str] = Field(default_factory=list, description="Organizzazioni menzionate")
    people: List[str] = Field(default_factory=list, description="Persone menzionate")
    locations: List[str] = Field(default_factory=list, description="Luoghi menzionati")
    dates: List[str] = Field(default_factory=list, description="Date menzionate")
    money: List[str] = Field(default_factory=list, description="Importi/valori monetari")
    cited_documents: List[str] = Field(default_factory=list, description="Documenti/allegati citati")


# Struttura vuota delle entita', usata come default e in caso di errore.
EMPTY_ENTITIES = {
    "organizations": [],
    "people": [],
    "locations": [],
    "dates": [],
    "money": [],
}


class SemanticEnricher:
    """Aggiunge summary e keywords a ogni sezione, se abilitato."""

    def __init__(self, client: Optional[Any], settings) -> None:
        """
        Inizializza l'arricchitore semantico.
        
        client: BackendLlmInterface (LLM backend usato per generare metadati).
        settings: config con flag enable_semantic_metadata e semantic_max_workers.
        """
        self.client = client
        self.settings = settings

    def _enabled(self) -> bool:
        """Vero solo se il flag e' attivo e ci sono client disponibile."""
        return (
            getattr(self.settings, "enable_semantic_metadata", False)
            and self.client is not None
        )

    def enrich(self, sections: list[Section]) -> list:
        """Riempie summary, keywords, topics ed entities nel metadata di ogni sezione."""
        if not self._enabled():
            return sections

        # Sequential execution: OpenAIChatCompletionClient is not thread-safe
        # when shared across threads. Execute enrichment one section at a time.
        logger.info(f"Semantic enrichment started for {len(sections)} sections (sequential mode)")
        for i, section in enumerate(sections, 1):
            try:
                self._enrich_one(section)
                logger.debug(f"[{i}/{len(sections)}] Enriched section")
            except Exception as e:
                logger.warning(f"[{i}/{len(sections)}] Failed to enrich section: {e}")
        return sections

    def _enrich_one(self, section: Section) -> None:
        """Arricchisce UNA sezione (1 chiamata LLM). Eseguita in parallelo."""
        text = self._section_text(section)
        if not text.strip():
            return
        data = self._summarize(text)
        section.metadata["summary"] = data.get("summary", "")
        section.metadata["keywords"] = data.get("keywords", [])
        section.metadata["topics"] = data.get("topics", [])
        section.metadata["entities"] = {
            "organizations": data.get("organizations", []),
            "people": data.get("people", []),
            "locations": data.get("locations", []),
            "dates": data.get("dates", []),
            "money": data.get("money", []),
        }
        section.metadata["cited_documents"] = data.get("cited_documents", [])

    def _section_text(self, section: Section) -> str:
        """Concatena il testo dei blocchi della sezione (limitato)."""
        parts = [block.content for block in section.blocks if block.content]
        return "\n".join(parts)[:6000]

    def _summarize(self, text: str) -> dict:
        """Chiama l'LLM e ritorna summary, keywords, topics, entities e cited_documents.

        Robusto agli errori: in caso di risposta non valida restituisce campi
        vuoti senza bloccare la pipeline.
        """
        try:
            prompt = USER_PROMPT_TEMPLATE.format(text=text)
            result = self.client.chat_structured(
                prompt=prompt,
                response_format=SemanticMetadata,
            )
            return {
                "summary": result.summary.strip(),
                "keywords": self._clean_list(result.keywords),
                "topics": self._clean_list(result.topics),
                "organizations": self._clean_list(result.organizations),
                "people": self._clean_list(result.people),
                "locations": self._clean_list(result.locations),
                "dates": self._clean_list(result.dates),
                "money": self._clean_list(result.money),
                "cited_documents": self._clean_list(result.cited_documents),
            }
        except Exception as error:
            logger.warning(f"Errore nella generazione metadati semantici: {error}")
            return {
                "summary": "",
                "keywords": [],
                "topics": [],
                "organizations": [],
                "people": [],
                "locations": [],
                "dates": [],
                "money": [],
                "cited_documents": [],
            }

    def _clean_list(self, values) -> list[str]:
        """Normalizza una lista di stringhe scartando i valori vuoti."""
        if not isinstance(values, list):
            return []
        return [str(v).strip() for v in values if str(v).strip()]

