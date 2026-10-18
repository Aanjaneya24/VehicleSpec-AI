"""Embedding, FAISS index construction, and persistent artifact management."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking import Chunk, chunk_pages
from src.config import Settings, settings
from src.pdf_ingest import extract_pdf_pages


class IndexingError(RuntimeError):
    pass


@lru_cache(maxsize=2)
def load_embedder(model_name: str):
    try:
        return SentenceTransformer(model_name)
    except Exception as error:
        raise IndexingError(f"Could not load embedding model '{model_name}': {error}") from error


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_index(pdf_path: str | Path, config: Settings = settings, progress=None) -> tuple[int, int]:
    pdf_path = Path(pdf_path)
    pages = extract_pdf_pages(pdf_path)
    if progress:
        progress(0.20, "Creating section-aware chunks…")
    chunks = chunk_pages(pages, config.chunk_size, config.chunk_overlap)
    if not chunks:
        raise IndexingError("No chunks could be created from extracted text.")
    if progress:
        progress(0.35, "Loading embedding model…")
    model = load_embedder(config.embedding_model)
    if progress:
        progress(0.45, "Generating embeddings…")
    vectors = model.encode([chunk.text for chunk in chunks], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")
    if vectors.ndim != 2 or not len(vectors):
        raise IndexingError("Embedding model returned invalid vectors.")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".faiss", dir=config.artifacts_dir, delete=False) as stream:
        temporary_index = Path(stream.name)
    try:
        faiss.write_index(index, str(temporary_index))
        os.replace(temporary_index, config.index_path)
    finally:
        temporary_index.unlink(missing_ok=True)
    metadata = "".join(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n" for chunk in chunks)
    _atomic_write_text(config.chunks_path, metadata)
    manifest = {
        "schema_version": 1,
        "source": pdf_path.name,
        "source_path": str(pdf_path.resolve()),
        "source_sha256": _file_sha256(pdf_path),
        "embedding_model": config.embedding_model,
        "dimension": int(vectors.shape[1]),
        "chunk_count": len(chunks),
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
    }
    _atomic_write_text(config.manifest_path, json.dumps(manifest, indent=2) + "\n")
    if progress:
        progress(1.0, "Index ready.")
    return len(pages), len(chunks)


def index_available(config: Settings = settings) -> bool:
    return config.index_path.is_file() and config.chunks_path.is_file() and config.manifest_path.is_file()
