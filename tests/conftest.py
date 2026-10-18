from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from src.config import settings


class FakeEmbedder:
    """Offline deterministic embedder that produces normalized two-dimensional vectors."""

    def encode(self, texts, **_kwargs):
        vectors = []
        for text in texts:
            normalized = text.lower()
            if "brake" in normalized or "caliper" in normalized:
                vector = [1.0, 0.0]
            elif "engine" in normalized:
                vector = [0.0, 1.0]
            else:
                vector = [0.5, 0.5]
            vectors.append(vector)
        vectors = np.asarray(vectors, dtype="float32")
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture
def fixture_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "fixture_manual.pdf"
    document = pymupdf.open()
    first = document.new_page()
    first.insert_text((72, 72), "SECTION 206-03: Front Disc Brake\n\nBrake caliper guide pin bolts\n37 Nm\n")
    document.new_page()  # intentionally blank
    third = document.new_page()
    third.insert_text((72, 72), "SECTION 303-01: Engine\n\nEngine mount bolts\n80 Nm\n")
    document.save(path)
    document.close()
    return path


@pytest.fixture
def test_settings(tmp_path: Path):
    return replace(settings, artifacts_dir=tmp_path / "artifacts")


@pytest.fixture
def fake_embedder(monkeypatch):
    model = FakeEmbedder()
    monkeypatch.setattr("src.indexing.load_embedder", lambda _name: model)
    monkeypatch.setattr("src.retrieval.load_embedder", lambda _name: model)
    return model
