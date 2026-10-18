"""Central configuration and artifact locations."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "220"))
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "6"))
    data_dir: Path = ROOT_DIR / "data"
    artifacts_dir: Path = ROOT_DIR / "artifacts"

    @property
    def default_pdf(self) -> Path:
        return self.data_dir / "manual.pdf"

    @property
    def index_path(self) -> Path:
        return self.artifacts_dir / "manual.faiss"

    @property
    def chunks_path(self) -> Path:
        return self.artifacts_dir / "chunks.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.artifacts_dir / "index_manifest.json"


settings = Settings()
