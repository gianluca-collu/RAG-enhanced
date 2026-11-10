from pathlib import Path

from .structure_analyzer import Section


class MetadataExtractor:
    """Arricchisce ogni Section con metadati generici e deterministici."""

    def extract(self, sections: list[Section], source) -> list[Section]:
        """Calcola i metadati documentali e strutturali di ogni sezione."""
        source = Path(source)
        file_type = source.suffix.lstrip(".").lower()

        for section in sections:
            # tipi dei blocchi normalizzati in minuscolo, come da desiderata
            # (es. ["table", "text"] invece di ["TABLE", "TEXT"]).
            types_upper = {block.type for block in section.blocks}
            content_types = sorted(t.lower() for t in types_upper)
            # il titolo del documento è il primo elemento del breadcrumb.
            # Ora è garantito da DocumentNormalizer che aggiunge il titolo come heading #
            document_title = section.breadcrumb[0] if section.breadcrumb else ""
            section.metadata = {
                "source_file": source.name,
                "file_type": file_type,
                "document_title": document_title,
                "section_path": " > ".join(section.breadcrumb),
                "section_depth": len(section.breadcrumb),
                "content_types": content_types,
                "contains_table": "TABLE" in types_upper,
                "contains_image": "IMAGE" in types_upper or "DIAGRAM" in types_upper,
            }
        return sections

