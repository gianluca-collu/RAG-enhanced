from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field


@dataclass
class Block:
    """Rappresenta un blocco di contenuto all'interno di una sezione."""
    type: str  # TEXT, TABLE, LIST, IMAGE, DIAGRAM, MIXED
    content: str
    page: int | None = None


@dataclass
class Section:
    """Rappresenta una sezione del documento con i suoi blocchi."""
    section_id: str
    doc_id: str
    breadcrumb: list[str]
    content_type: str = "TEXT"
    page: int | None = None
    blocks: list[Block] = field(default_factory=list)
    # Stage 6: metadati generici deterministici (riempiti da MetadataExtractor).
    metadata: dict = field(default_factory=dict)


class StructureAnalyzer:
    """Analizza la struttura del Markdown e crea sezioni."""

    def _split_by_heading(self, markdown: str) -> list[tuple[int, str, str]]:
        """Divide il Markdown in sezioni usando gli heading (#, ##, ###...).
           Per ogni heading estrae: livello, titolo e corpo della sezione.
           Il corpo va dall'heading corrente fino al prossimo heading.
           Esempio input:
           "# Titolo\nTesto\n## Dettaglio\nAltro testo"
           Esempio output:
           [(1, "Titolo", "Testo"), (2, "Dettaglio", "Altro testo")]"""

        pattern = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
        matches = list(pattern.finditer(markdown))
        parts = []

        for index, match in enumerate(matches):
            # Il corpo della sezione va dal fine dell'heading al prossimo heading
            end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)

            parts.append((
                len(match.group(1)),           # level (numero di #)
                match.group(2).strip(),        # title
                markdown[match.end():end].strip(),  # body
            ))

        return parts

    def _heading_number(self, title: str) -> str | None:
        """Numero gerarchico iniziale del titolo, es. '16.2'.
        Match solo se il numero è seguito da spazio, punto, parentesi o fine:
        così '10113RE' (codice documento) NON viene scambiato per un numero."""
        match = re.match(r"^(\d+(?:\.\d+)*)(?=[\s.)]|$)", title.strip())
        return match.group(1) if match else None


    def make_create_breadcrumb(self, document_splitted, doc_id):
        sections = []
        doc_title = document_splitted[0][1] if document_splitted else ""
        numbered = {}                  # "28" -> "28. Segnalazioni..."
        last_crumb = [doc_title]       # percorso dell'ultimo capitolo numerato visto
        page = 1

        for level, title, body in document_splitted:
            num = self._heading_number(title)

            if num:
                # Titolo numerato: ricostruisce il percorso dai prefissi numerici.
                numbered[num] = title
                crumb = [doc_title]
                parts = num.split(".")
                for i in range(1, len(parts) + 1):
                    prefix = ".".join(parts[:i])
                    if prefix in numbered:
                        crumb.append(numbered[prefix])
                headings = crumb
                last_crumb = crumb                    # memorizza il capitolo padre
            elif title == doc_title:
                headings = [doc_title]
            else:
                # Titolo NON numerato: eredita il capitolo numerato precedente.
                headings = last_crumb + [title]

            section = Section(
                section_id=self._section_id(doc_id, headings),
                doc_id=doc_id,
                breadcrumb=headings.copy(),
                page=page,
            )
            section.content_type = self._section_type(body)
            section.blocks = self._parse_blocks(body, page)
            sections.append(section)
            page += len(re.findall(r"<!--\s*PageBreak\s*-->", body, re.IGNORECASE))

        return sections


    def _section_id(self, doc_id: str, breadcrumb: list[str]) -> str:
            """Genera ID stabile per una sezione usando doc_id e breadcrumb."""
            value = f"{doc_id}:{'/'.join(breadcrumb)}"
            return hashlib.sha1(value.encode()).hexdigest()

    def _parse_blocks(self, content: str,page: int | None = None) -> list[Block]:
        """Scompone una sezione solo quando contiene """
        blocks = []
        current_type = None
        current_content = []
        current_page = page

        def flush_current_block():
            nonlocal current_type, current_content
            if current_content:
                blocks.append(Block(
                    type=current_type,
                    content="\n".join(current_content),
                    page=current_page,
                ))
                current_type = None
                current_content = []

        for line in content.splitlines():
            cleaned = line.strip()

            if not cleaned:
                continue

            if re.fullmatch(r"<!--\s*PageBreak\s*-->", cleaned, re.IGNORECASE):
                flush_current_block()
                current_page = current_page + 1 if current_page is not None else None
                continue

            line_type = self._line_type(cleaned)

            if current_content and line_type != current_type:
                flush_current_block()

            current_type = line_type
            current_content.append(cleaned)

        flush_current_block()

        return blocks




    def _section_type(self, content: str) -> str:
        """Classifica l'intera sezione come TEXT o MIXED."""

        types = {
            self._line_type(line.strip())
            for line in content.splitlines()
            if line.strip()
        }

        return "MIXED" if len(types) > 1 else next(iter(types), "TEXT")


    def _line_type(self, line: str) -> str:
        """Riconosce il tipo di una singola riga Markdown."""

        line = line.strip()

        if line.startswith("|") and line.endswith("|"):
            return "TABLE"

        if "![" in line and "]" in line:
            return "IMAGE"

        if re.match(r"^(?:[-*+•▪➢]|\d+[.)])\s+", line):
            return "LIST"

        return "TEXT"

    def analyze(self, markdown: str, doc_id: str, di_result = None) -> list[Section]:
            """Ricostruisce sezioni e sottosezioni dagli heading Markdown."""
            #1 divide il documento markdown per heading
            document_splitted = self._split_by_heading(markdown)

            # Se non ci sono heading, usa l'intero markdown come una sezione di default
            if not document_splitted and markdown.strip():
                document_splitted = [(1, "Contenuto", markdown)]

            #2 aggiunge le sezioni al documento precedente
            docuemnt_splitted_enriched = self.make_create_breadcrumb(document_splitted, doc_id)
            return docuemnt_splitted_enriched


