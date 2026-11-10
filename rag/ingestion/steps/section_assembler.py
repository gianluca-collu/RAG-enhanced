from .structure_analyzer import Block, Section


class SectionAssembler:
    """Aggrega blocchi coerenti dentro ogni sezione."""

    def assemble(self, sections: list[Section]) -> list[Section]:
        """Ritorna le sezioni con i frammenti di tabella fusi."""
        for section in sections:
            section.blocks = self._merge_blocks(section.blocks)
        return sections

    def _merge_blocks(self, blocks: list[Block]) -> list[Block]:
        """Fonde i frammenti di tabella (stesso numero di colonne) nella
        sezione, anche se separati da immagini o header di pagina."""
        merged: list[Block] = []
        table_by_cols: dict[int, Block] = {}   # n_colonne -> tabella gia' in merged

        for block in blocks:
            if block.type == "TABLE":
                cols = self._cols(block)
                if cols in table_by_cols:
                    # Frammento della stessa tabella: appendi al blocco esistente.
                    table_by_cols[cols].content += "\n" + block.content
                else:
                    new_block = Block(block.type, block.content, block.page)
                    merged.append(new_block)
                    table_by_cols[cols] = new_block
            else:
                merged.append(block)

        return merged

    def _cols(self, block: Block) -> int:
        """Numero di colonne di una tabella = numero di '|' nella prima riga."""
        for line in block.content.splitlines():
            if line.strip().startswith("|"):
                return line.count("|")
        return 0

