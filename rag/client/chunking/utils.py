import io
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import openpyxl
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)


def generate_kb_report(settings, kb_index: str = None) -> Path:
    """
    Generate an Excel report of indexed documents and save it.

    Includes:
    - Document title, chunk count, content types
    - Pages, sections, summaries, keywords
    - Totals by type

    Args:
        settings: Settings object with search credentials
        kb_index: Optional KB index name; uses settings.search_index_name if not provided

    Returns:
        Path to the generated Excel file
    """
    # Create search client
    index_name = kb_index or settings.search_index_name
    client = SearchClient(
        endpoint=settings.search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(settings.search_key),
    )

    def new_doc():
        return {
            "chunks": 0,
            "TEXT": 0,
            "TABLE": 0,
            "IMAGE": 0,
            "pages": set(),
            "sections": set(),
            "with_summary": 0,
            "with_keywords": 0,
            "keywords": set(),
        }

    stats = defaultdict(new_doc)

    # Azure SDK handles pagination automatically
    # Query only the fields that exist in the llama-index schema
    results = client.search(
        search_text="*",
        select=["chunk_id", "chunk", "metadata", "doc_id"],
    )

    # Process all chunks
    import json
    for d in results:
        doc_id = d.get("doc_id")
        s = stats[doc_id]
        s["chunks"] += 1
        
        # Metadata is stored as a JSON string in the new schema
        try:
            metadata = json.loads(d.get("metadata") or "{}")
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        
        # Extract metadata fields (if they exist)
        ct = metadata.get("content_type", "TEXT")
        s[ct] = s.get(ct, 0) + 1
        
        page = metadata.get("page")
        if page is not None:
            s["pages"].add(page)
        
        section = metadata.get("section_path")
        if section:
            s["sections"].add(section)
        
        summary = metadata.get("summary", "")
        if (summary or "").strip():
            s["with_summary"] += 1
        
        kws = metadata.get("keywords") or []
        if kws:
            s["with_keywords"] += 1
            if isinstance(kws, list):
                s["keywords"].update(kws)
            elif isinstance(kws, str):
                s["keywords"].add(kws)

    # Build Excel workbook
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Riepilogo"

    # Header row
    ws.append(
        [
            "Documento",
            "Chunk totali",
            "TEXT",
            "TABLE",
            "IMAGE",
            "Pagine",
            "Sezioni distinte",
            "Chunk con summary",
            "Chunk con keywords",
            "Keywords principali",
        ]
    )

    # Data rows
    for doc, s in sorted(stats.items()):
        top_kw = ", ".join(sorted(s["keywords"])[:15])
        ws.append(
            [
                doc,
                s["chunks"],
                s["TEXT"],
                s["TABLE"],
                s["IMAGE"],
                len(s["pages"]),
                len(s["sections"]),
                s["with_summary"],
                s["with_keywords"],
                top_kw,
            ]
        )

    # Totals row
    ws.append([])
    ws.append(
        [
            "TOTALE",
            sum(s["chunks"] for s in stats.values()),
            sum(s["TEXT"] for s in stats.values()),
            sum(s["TABLE"] for s in stats.values()),
            sum(s["IMAGE"] for s in stats.values()),
            "",
            "",
            "",
            "",
            "",
        ]
    )

    # Save to local file
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"kb_report_{timestamp}.xlsx"

    wb.save(report_path)
    logger.info("KB report saved: %s", report_path)

    return report_path

