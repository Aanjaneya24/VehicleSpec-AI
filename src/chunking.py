"""Deterministic, section-aware chunk creation."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from hashlib import sha256

from src.pdf_ingest import PageText

SECTION_PATTERN = re.compile(r"^(?:SECTION\s+)?\d{2,3}-\d{2}(?:-\d{2})?(?::.*)?$", re.IGNORECASE)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    page: int
    section: str
    text: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _section_for(lines: list[str], current: str) -> str:
    for line in lines:
        if SECTION_PATTERN.match(line.strip()):
            current = line.strip()
    return current


def chunk_pages(pages: list[PageText], max_chars: int = 1500, overlap: int = 220) -> list[Chunk]:
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be non-negative and smaller than max_chars")
    chunks: list[Chunk] = []
    section = ""
    for page in pages:
        blocks = [block.strip() for block in re.split(r"\n\s*\n", page.text) if block.strip()]
        if not blocks:
            blocks = [page.text]
        buffer = ""
        buffer_section = section
        for block in blocks:
            lines = block.splitlines()
            next_section = _section_for(lines, section)
            # A heading starts a new context; never assign it retroactively to a
            # preceding buffered table/paragraph.
            if next_section != section and buffer:
                chunks.append(_make_chunk(page, buffer_section, buffer, len(chunks)))
                buffer = ""
            section = next_section
            block_section = section
            candidate = block if not buffer else f"{buffer}\n{block}"
            if len(candidate) <= max_chars:
                buffer = candidate
                buffer_section = block_section
                continue
            if buffer:
                # Keep a stand-alone heading with the following table/paragraph.
                if buffer.strip() == block_section:
                    block = f"{buffer}\n{block}"
                    buffer = ""
                else:
                    chunks.append(_make_chunk(page, buffer_section, buffer, len(chunks)))
            # Split an oversized table/paragraph only on line boundaries.
            while len(block) > max_chars:
                split_at = block.rfind("\n", 0, max_chars)
                if split_at <= 0:
                    split_at = max_chars
                part = block[:split_at].strip()
                chunks.append(_make_chunk(page, block_section, part, len(chunks)))
                tail = part[-overlap:] if overlap else ""
                block = (tail + "\n" + block[split_at:]).strip()
            buffer = block
            buffer_section = block_section
        if buffer:
            chunks.append(_make_chunk(page, buffer_section, buffer, len(chunks)))
    return chunks


def _make_chunk(page: PageText, section: str, text: str, position: int) -> Chunk:
    stable = f"{page.source}|{page.page}|{position}|{text}".encode("utf-8")
    return Chunk(sha256(stable).hexdigest()[:16], page.source, page.page, section, text)
