import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from openpyxl import Workbook, load_workbook

logger = logging.getLogger(__name__)

CONFIG_FILE = Path(__file__).with_name("retrieval_debug.json")
OUTPUT_FILE = Path(__file__).resolve().parents[2] / "output" / "retrieval_chunks.xlsx"
HEADERS = (
    "question",
    "rank",
    "document",
    "section_path",
    "page",
    "similarity_score",
    "chunk_text",
    "timestamp_utc",
)
EXCEL_CELL_LIMIT = 32767
_EXPORT_LOCK = Lock()


def _safe_text(value) -> str:
    text = "" if value is None else str(value)
    if text.lstrip().startswith(("=", "+", "-", "@")):
        text = "'" + text
    return text[:EXCEL_CELL_LIMIT]


def _metadata(source: dict) -> dict:
    metadata = source.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
    return metadata if isinstance(metadata, dict) else {}


def export_retrieval(question: str, sources: list[dict]) -> None:
    try:
        settings = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot read retrieval debug settings (%s)", type(exc).__name__)
        return

    if settings.get("enabled") is not True:
        return

    try:
        with _EXPORT_LOCK:
            OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            if OUTPUT_FILE.exists():
                workbook = load_workbook(OUTPUT_FILE)
                worksheet = workbook["Retrieval"] if "Retrieval" in workbook.sheetnames else workbook.active
            else:
                workbook = Workbook()
                worksheet = workbook.active
                worksheet.title = "Retrieval"

            if worksheet.max_row == 1 and worksheet["A1"].value is None:
                for column, header in enumerate(HEADERS, start=1):
                    worksheet.cell(row=1, column=column, value=header)
                worksheet.freeze_panes = "A2"

            timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            ranked_sources = list(enumerate(sources, start=1)) or [(0, {})]
            for rank, source in ranked_sources:
                metadata = _metadata(source)
                document = (
                    metadata.get("source")
                    or metadata.get("source_file")
                    or metadata.get("document_title")
                    or metadata.get("doc_id")
                )
                worksheet.append((
                    _safe_text(question),
                    rank,
                    _safe_text(document),
                    _safe_text(metadata.get("section_path")),
                    metadata.get("page"),
                    source.get("score"),
                    _safe_text(source.get("content")),
                    timestamp,
                ))

            worksheet.auto_filter.ref = worksheet.dimensions
            workbook.save(OUTPUT_FILE)
            workbook.close()
    except Exception as exc:
        logger.warning("Cannot write retrieval debug workbook (%s)", type(exc).__name__)
