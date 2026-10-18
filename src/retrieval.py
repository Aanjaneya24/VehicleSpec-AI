"""Hybrid semantic + lexical retrieval over persisted chunk artifacts."""
from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from dataclasses import dataclass

import faiss
import numpy as np

from src.config import Settings, settings
from src.indexing import load_embedder


class RetrievalError(RuntimeError):
    pass


@dataclass(frozen=True)
class RetrievedPassage:
    chunk_id: str
    source: str
    page: int
    section: str
    text: str
    semantic_score: float
    lexical_score: float
    score: float


def index_status(config: Settings = settings) -> tuple[bool, str]:
    """Fast status check: read only the manifest and file sizes; do not load FAISS or the embedder."""
    required = (config.index_path, config.chunks_path, config.manifest_path)
    if not all(path.is_file() for path in required):
        return False, "No complete index is available. Build the manual index first."
    try:
        manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("embedding_model") != config.embedding_model:
            return False, "Index configuration is incompatible with current settings. Rebuild the index."
        indexed_source = Path(str(manifest.get("source_path", "")))
        if indexed_source.is_file():
            digest = sha256(indexed_source.read_bytes()).hexdigest()
            if digest != manifest.get("source_sha256"):
                return False, "The indexed PDF has changed. Rebuild the index."
    except Exception:
        return False, "Index artifacts cannot be read. Rebuild the index."
    return True, f"Index ready: {manifest.get('chunk_count', '?')} chunks from {manifest.get('source', 'unknown')}."


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9-]*", value.lower()))


def retrieve(query: str, top_k: int = 6, config: Settings = settings) -> list[RetrievedPassage]:
    query = query.strip()
    if not query:
        raise RetrievalError("Enter a question before searching.")
    if not 1 <= top_k <= 50:
        raise RetrievalError("top_k must be between 1 and 50.")
    usable, message = index_status(config)
    if not usable:
        raise RetrievalError(message)
    try:
        metadata = [json.loads(line) for line in config.chunks_path.read_text(encoding="utf-8").splitlines() if line]
        index = faiss.read_index(str(config.index_path))
    except Exception as error:
        raise RetrievalError(f"Could not load index artifacts: {error}") from error
    # Full consistency check: chunk count must match FAISS ntotal and the manifest; all metadata fields must be valid.
    try:
        manifest_count = json.loads(config.manifest_path.read_text(encoding="utf-8")).get("chunk_count")
    except Exception as error:
        raise RetrievalError(f"Could not read index manifest: {error}") from error
    if index.ntotal != len(metadata) or index.ntotal != manifest_count:
        raise RetrievalError("Index artifacts are inconsistent. Rebuild the index.")
    for item in metadata:
        if not {"chunk_id", "source", "page", "section", "text"} <= item.keys() or not isinstance(item["page"], int):
            raise RetrievalError("Index artifacts are inconsistent. Rebuild the index.")
    model = load_embedder(config.embedding_model)
    vector = np.asarray(model.encode([query], convert_to_numpy=True, normalize_embeddings=True), dtype="float32")
    if vector.shape[1] != index.d:
        raise RetrievalError("Embedding dimensions do not match the saved index. Rebuild it with the current model.")
    candidate_count = min(len(metadata), max(top_k * 4, top_k))
    distances, positions = index.search(vector, candidate_count)
    query_tokens = _tokens(query)
    passages = []
    for semantic, position in zip(distances[0], positions[0]):
        if position < 0:
            continue
        item = metadata[int(position)]
        lexical = len(query_tokens & _tokens(item["text"])) / max(len(query_tokens), 1)
        score = 0.85 * float(semantic) + 0.15 * lexical
        passages.append(RetrievedPassage(**item, semantic_score=float(semantic), lexical_score=lexical, score=score))
    return sorted(passages, key=lambda item: item.score, reverse=True)[:top_k]
