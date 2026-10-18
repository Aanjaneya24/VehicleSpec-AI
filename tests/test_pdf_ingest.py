import pytest

from src.pdf_ingest import PDFIngestError, extract_pdf_pages


def test_extract_pdf_preserves_filename_one_based_pages_and_skips_blank(fixture_pdf):
    pages = extract_pdf_pages(fixture_pdf)
    assert [(page.source, page.page) for page in pages] == [("fixture_manual.pdf", 1), ("fixture_manual.pdf", 3)]
    assert "Brake caliper guide pin bolts" in pages[0].text


def test_extract_pdf_rejects_missing_and_non_pdf_files(tmp_path):
    with pytest.raises(PDFIngestError, match="not found"):
        extract_pdf_pages(tmp_path / "missing.pdf")
    non_pdf = tmp_path / "manual.txt"
    non_pdf.write_text("just text", encoding="utf-8")
    with pytest.raises(PDFIngestError, match="not a PDF"):
        extract_pdf_pages(non_pdf)
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(PDFIngestError, match="empty"):
        extract_pdf_pages(empty)
    broken = tmp_path / "broken.pdf"
    broken.write_text("not a PDF", encoding="utf-8")
    with pytest.raises(PDFIngestError, match="Unable to read"):
        extract_pdf_pages(broken)
