from src.chunking import chunk_pages
from src.pdf_ingest import PageText, clean_text


def test_clean_text_preserves_line_layout():
    assert clean_text("  Brake\tcaliper \n\n 37   Nm ") == "Brake caliper\n37 Nm"


def test_chunking_preserves_required_metadata_and_is_deterministic():
    pages = [PageText("manual.pdf", 17, "SECTION 206-03: Front Disc Brake\n\nBrake caliper guide pin bolts\n37 Nm")]
    first = chunk_pages(pages, max_chars=200, overlap=20)
    assert first == chunk_pages(pages, max_chars=200, overlap=20)
    assert first[0].source == "manual.pdf"
    assert first[0].page == 17
    assert first[0].section == "SECTION 206-03: Front Disc Brake"
    assert "37 Nm" in first[0].text


def test_chunking_does_not_apply_a_later_section_to_preceding_content():
    preliminary = "Preliminary torque table " + ("data " * 20) + "37 Nm"
    pages = [PageText("manual.pdf", 1, f"{preliminary}\n\nSECTION 206-03: Brakes\n\nGuide pin bolts\n42 Nm")]
    chunks = chunk_pages(pages, max_chars=100, overlap=10)
    assert chunks[0].section == ""
    assert chunks[-1].section == "SECTION 206-03: Brakes"


def test_oversized_table_uses_overlap_and_keeps_values():
    table = "SECTION 206-03: Brakes\n\n" + "\n".join(f"Bolt {number} {number} Nm" for number in range(25))
    chunks = chunk_pages([PageText("manual.pdf", 4, table)], max_chars=100, overlap=20)
    assert len(chunks) > 1
    assert all(chunk.section == "SECTION 206-03: Brakes" for chunk in chunks)
    assert "Bolt 0 0 Nm" in chunks[0].text
    assert any("Bolt 24 24 Nm" in chunk.text for chunk in chunks)
