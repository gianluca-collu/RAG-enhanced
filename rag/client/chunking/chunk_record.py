def chunk_to_record(chunk, source: str = "") -> dict:
    """Trasforma un oggetto Chunk nel dizionario usato da export e indexing.

    I metadati semantici (summary/keywords/topics) e documentali
    (document_title/section_path/content_types) sono promossi a top-level per
    comodita' in Azure AI Search, ma restano anche dentro 'metadata'.
    """
    meta = chunk.metadata or {}
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "section_id": chunk.section_id,
        "breadcrumb": chunk.breadcrumb,
        "content": chunk.content,
        "contextualized_content": chunk.contextualized_content,
        "content_type": chunk.content_type,
        "page": chunk.page,
        "token_count": chunk.token_count,
        "document_title": meta.get("document_title", ""),
        "section_path": meta.get("section_path", ""),
        "content_types": meta.get("content_types", []),
        "summary": meta.get("summary", ""),
        "keywords": meta.get("keywords", []),
        "topics": meta.get("topics", []),
        "metadata": chunk.metadata,
        "source": source,
    }

