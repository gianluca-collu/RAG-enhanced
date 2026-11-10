from __future__ import annotations

import logging
import re
import hashlib
from dataclasses import dataclass

import tiktoken
from llama_index.core.node_parser import TokenTextSplitter

from .structure_analyzer import Block, Section

logger = logging.getLogger(__name__)

# Numero minimo di token "reali" perche' un chunk di testo sia informativo.
MIN_CONTENT_TOKENS = 15

# Riga separatrice di una tabella markdown, es. "| --- | --- |".
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}")


@dataclass
class Chunk:
    """Rappresenta un chunk di contenuto finale per la KB."""
    chunk_id: str
    doc_id: str
    section_id: str
    breadcrumb: list[str]
    content: str                 # testo pulito del chunk (senza breadcrumb)
    contextualized_content: str  # Stage 8: breadcrumb + testo, cio' che si vettorizza
    content_type: str
    page: int | None
    token_count: int
    metadata: dict               # Stage 6: metadati generici della sezione


class ChunkAssembler:
    """Assembla i chunk finali dalle sezioni processate."""

    def __init__(
        self,
        max_tokens: int = 1024,
        overlap: int = 128,
        table_max_tokens: int = 800,
    ) -> None:
        """Inizializza splitter, tokenizer e soglia di split delle tabelle."""
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.splitter = TokenTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=overlap,
        )
        self.table_max_tokens = table_max_tokens

    def assemble(self, sections: list[Section]) -> list[Chunk]:
        """Trasforma le sezioni processate in chunk finali."""
        chunks = []
        for section in sections:
            for block in section.blocks:
                contents = self._split_block(block)
                chunks.extend(self._make_chunks(section, block, contents))
        return self._dedup(chunks)

    def _split_block(self, block: Block) -> list[str]:
        """Sceglie la strategia di split in base al tipo di blocco (Stage 7).

        - IMAGE, DIAGRAM, LIST: sempre atomici (spezzarli perde significato);
        - TABLE: atomica se piccola, altrimenti divisa in sotto-tabelle con
          intestazione ripetuta;
        - TEXT: spezzato con finestra a token.
        """
        if block.type in {"IMAGE", "DIAGRAM", "LIST"}:
            return [block.content]
        if block.type == "TABLE":
            return self._split_table(block.content)
        return self.splitter.split_text(block.content)

    def _split_table(self, content: str) -> list[str]:
        """Divide una tabella markdown grande ripetendo l'header su ogni pezzo.

        Se la tabella sta sotto la soglia resta intera. Se ha una riga di
        intestazione (con separatore '| --- |') quella viene ripetuta; se non
        ce l'ha, si ripete comunque la prima riga come intestazione, cosi'
        ogni sotto-tabella conserva le colonne di riferimento.
        """
        if self._count(content) <= self.table_max_tokens:
            return [content]

        header, body = self._table_header_body(content.splitlines())
        header_text = "\n".join(header)
        budget = self.table_max_tokens - self._count(header_text)

        groups: list[list[str]] = []
        current: list[str] = []
        used = 0
        for row in body:
            row_cost = self._count(row) + 1
            if current and used + row_cost > budget:
                groups.append(current)
                current, used = [], 0
            current.append(row)
            used += row_cost
        if current:
            groups.append(current)

        parts = [header_text + "\n" + "\n".join(rows) for rows in groups]
        logger.debug(
            "Tabella grande divisa in %d sotto-tabelle (header ripetuto)",
            len(parts),
        )
        return parts

    def _table_header_body(self, lines: list[str]) -> tuple[list[str], list[str]]:
        """Ritorna (righe di header, righe di corpo) di una tabella markdown."""
        if len(lines) >= 2 and _TABLE_SEPARATOR.match(lines[1]):
            return lines[:2], lines[2:]        # header vero + separatore
        return lines[:1], lines[1:]            # nessun header: usa la prima riga

    def _count(self, text: str) -> int:
        """Numero di token del testo."""
        return len(self.encoder.encode(text))

    def _clean_content(self, text: str) -> str:
        """Rimuove marker DI e OCR grezzo delle figure dal contenuto."""
        # Contenuto delle figure: la semantica e' nel blocco IMAGE descritto.
        text = re.sub(r"<figure\b[^>]*>.*?</figure>", "", text,
                      flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"</?figure\b[^>]*>", "", text, flags=re.IGNORECASE)
        # Commenti DI: PageBreak, PageHeader, PageFooter, PageNumber.
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        # Normalizza gli a-capo multipli lasciati dalle rimozioni.
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _is_informative(self, text: str) -> bool:
        """Vero se, dopo la pulizia, resta contenuto sopra la soglia minima."""
        return self._count(text) >= MIN_CONTENT_TOKENS

    def _context_prefix(self, section: Section) -> str:
        """Prefisso di contesto Stage 8: 'Documento: X. Sezione: Y.'"""
        bc = section.breadcrumb
        if not bc:
            return ""
        doc = bc[0]
        if len(bc) > 1:
            return f"Documento: {doc}. Sezione: {' > '.join(bc[1:])}.\n\n"
        return f"Documento: {doc}.\n\n"

    def _make_chunks(
        self,
        section: Section,
        block: Block,
        contents: list[str],
    ) -> list[Chunk]:
        """Crea chunk puliti con ID, contesto, tipo e conteggio token."""
        prefix = self._context_prefix(section)

        chunks = []
        for index, raw in enumerate(contents):
            content = self._clean_content(raw)

            # Scarta i chunk di testo non informativi (tabelle/immagini restano).
            if block.type == "TEXT" and not self._is_informative(content):
                logger.debug(
                    "Chunk di testo scartato (%d token < %d) in sezione %s",
                    self._count(content), MIN_CONTENT_TOKENS, section.section_id,
                )
                continue

            contextualized = prefix + content
            key = f"{section.section_id}:{block.type}:{index}:{content}"
            chunks.append(Chunk(
                chunk_id=hashlib.sha1(key.encode()).hexdigest(),
                doc_id=section.doc_id,
                section_id=section.section_id,
                breadcrumb=section.breadcrumb,
                content=content,
                contextualized_content=contextualized,
                content_type=block.type,
                page=block.page,
                token_count=self._count(contextualized),
                metadata=dict(section.metadata),
            ))
        return chunks

    def _dedup(self, chunks: list[Chunk]) -> list[Chunk]:
        """Rimuove i quasi-duplicati da overlap, solo entro la stessa sezione.

        Chiave = (section_id, fingerprint): l'overlap che genera i doppioni e'
        sempre intra-sezione, quindi contenuti identici in sezioni diverse
        (es. una tabella riepilogo ripetuta) vengono conservati. Si tiene il
        primo."""
        seen: set[tuple[str, str]] = set()
        unique: list[Chunk] = []
        for chunk in chunks:
            fingerprint = re.sub(r"\W+", "", chunk.contextualized_content.lower())
            key = (chunk.section_id, fingerprint)
            if key in seen:
                continue
            seen.add(key)
            unique.append(chunk)
        return unique

