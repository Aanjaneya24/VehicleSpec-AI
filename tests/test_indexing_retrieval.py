import json

import faiss
import numpy as np
import pytest

from src.indexing import build_index, index_available
from src.retrieval import RetrievalError, index_status, retrieve


def test_build_persist_reload_and_retrieve_fixture_pdf(fixture_pdf, test_settings, fake_embedder):
    pages, chunks = build_index(fixture_pdf, config=test_settings)
    assert (pages, chunks) == (2, 2)
    assert index_available(test_settings)
    manifest = json.loads(test_settings.manifest_path.read_text(encoding="utf-8"))
    assert manifest["chunk_count"] == 2
    assert manifest["dimension"] == 2
    assert np.isclose(np.linalg.norm(faiss.read_index(str(test_settings.index_path)).reconstruct(0)), 1.0)
    ok, msg = index_status(test_settings)
    assert ok
    assert msg.startswith("Index ready:")
    passages = retrieve("brake caliper guide pin torque", top_k=2, config=test_settings)
    assert passages[0].page == 1
    assert "37 Nm" in passages[0].text
    assert passages[0].score >= passages[1].score


def test_retrieval_detects_inconsistent_metadata_and_model_config(fixture_pdf, test_settings, fake_embedder):
    build_index(fixture_pdf, config=test_settings)
    test_settings.chunks_path.write_text(test_settings.chunks_path.read_text(encoding="utf-8") + '{"bad": true}\n', encoding="utf-8")
    with pytest.raises(RetrievalError, match="inconsistent"):
        retrieve("brake", config=test_settings)


def test_retrieval_detects_changed_source_pdf(fixture_pdf, test_settings, fake_embedder):
    build_index(fixture_pdf, config=test_settings)
    fixture_pdf.write_bytes(fixture_pdf.read_bytes() + b"changed")
    with pytest.raises(RetrievalError, match="PDF has changed"):
        retrieve("brake", config=test_settings)


def test_retrieval_rejects_empty_query_and_missing_artifacts(test_settings):
    with pytest.raises(RetrievalError, match="Enter a question"):
        retrieve(" ", config=test_settings)
    with pytest.raises(RetrievalError, match="No complete index"):
        retrieve("brake", config=test_settings)


def test_build_index_rejects_missing_and_invalid_pdf(tmp_path, test_settings, fake_embedder):
    from src.pdf_ingest import PDFIngestError

    with pytest.raises(PDFIngestError, match="not found"):
        build_index(tmp_path / "nonexistent.pdf", config=test_settings)
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("hello", encoding="utf-8")
    with pytest.raises(PDFIngestError, match="not a PDF"):
        build_index(txt_file, config=test_settings)

