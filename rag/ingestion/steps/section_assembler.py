from .structure_analyzer import Block, Section


class SectionAssembler:
    """Aggrega blocchi coerenti dentro ogni sezione."""

    def assemble(self, sections: list[Section]) -> list[Section]:
        """Ritorna le sezioni con i frammenti di tabella fusi."""
        for section in sections:
            section.blocks = self._merge_blocks(section.blocks)
        return sections

    def _merge_blocks(self, blocks: list[Block]) -> list[Block]:
        """Merge only adjacent table fragments with matching columns and headers."""
        merged: list[Block] = []

        for block in blocks:
            if (
                block.type == "TABLE"
                and merged
                and merged[-1].type == "TABLE"
                and self._cols(merged[-1]) == self._cols(block)
                and self._header(merged[-1]) == self._header(block)
            ):
                merged[-1].content += "\n" + block.content
            else:
                merged.append(Block(block.type, block.content, block.page))

        return merged

    def _header(self, block: Block) -> str:
        for line in block.content.splitlines():
            if line.strip() and not line.strip().startswith("| ---"):
                return " ".join(line.split()).casefold()
        return ""

    def _cols(self, block: Block) -> int:
        """Numero di colonne di una tabella = numero di '|' nella prima riga."""
        for line in block.content.splitlines():
            if line.strip().startswith("|"):
                return line.count("|")
        return 0

