"""Build the persisted manual index without starting the Streamlit UI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import settings
from src.indexing import build_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a VehicleSpec AI FAISS index.")
    parser.add_argument("pdf", nargs="?", type=Path, default=settings.default_pdf, help="Path to the workshop-manual PDF")
    args = parser.parse_args()
    try:
        pages, chunks = build_index(args.pdf, progress=lambda _, message: print(message))
    except Exception as error:
        print(f"Index build failed: {error}", file=sys.stderr)
        return 1
    print(f"Indexed {pages} text-bearing pages into {chunks} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

