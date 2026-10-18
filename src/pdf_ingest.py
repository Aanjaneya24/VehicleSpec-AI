"""Text-first PDF ingestion with stable, one-based page metadata."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz


class PDFIngestError(ValueError):
    """Raised when a PDF cannot produce usable text."""


@dataclass(frozen=True)
class PageText:
    source: str
    page: int
    text: str


def clean_text(text: str) -> str:
    """Remove noise while retaining newlines that convey table/paragraph layout."""
    lines = []
    for line in text.replace("\xa0", " ").splitlines():
        normalized = re.sub(r"[ \t]+", " ", line).strip()
        if normalized:
            lines.append(normalized)
    return "\n".join(lines)


def extract_pdf_pages(pdf_path: str | Path) -> list[PageText]:
    path = Path(pdf_path)
    if not path.is_file():
        raise PDFIngestError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise PDFIngestError(f"File is not a PDF: {path.name}")
    if path.stat().st_size == 0:
        raise PDFIngestError(f"The PDF file '{path.name}' is empty (0 bytes).")
    try:
        document = fitz.open(path)
    except Exception as error:
        raise PDFIngestError(f"Unable to read PDF: {error}") from error

    pages: list[PageText] = []
    try:
        for number, page in enumerate(document, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append(PageText(source=path.name, page=number, text=text))
    finally:
        document.close()
    if not pages:
        raise PDFIngestError("No selectable text was found in this PDF. Scanned PDFs require OCR, which is out of scope.")
    return pages
