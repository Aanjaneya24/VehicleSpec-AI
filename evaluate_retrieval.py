"""Evaluate retrieval against manually verified source-page labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.retrieval import RetrievalError, retrieve


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure retrieval Hit@k and MRR using verified manual labels.")
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/manual_verified_cases.json"))
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    try:
        cases = json.loads(args.dataset.read_text(encoding="utf-8"))["cases"]
    except (OSError, ValueError, KeyError) as error:
        print(f"Could not read evaluation dataset: {error}", file=sys.stderr)
        return 1
    verified = [case for case in cases if case.get("verification_status") == "verified" and not case.get("expected_not_found", False)]
    if not verified:
        print("No manually verified cases are available; no retrieval metrics were calculated.")
        return 0
    hits = 0
    reciprocal_ranks: list[float] = []
    for case in verified:
        expected = {(item["source"], item["page"]) for item in case.get("expected_results", [])}
        try:
            passages = retrieve(case["question"], args.top_k)
        except RetrievalError as error:
            print(f"Retrieval failed for {case.get('id', 'unknown')}: {error}", file=sys.stderr)
            return 1
        ranks = [rank for rank, passage in enumerate(passages, 1) if (passage.source, passage.page) in expected]
        if ranks:
            hits += 1
            reciprocal_ranks.append(1 / min(ranks))
        else:
            reciprocal_ranks.append(0.0)
    unsupported = sum(case.get("verification_status") == "verified" and case.get("expected_not_found", False) for case in cases)
    print(f"Verified retrieval cases: {len(verified)}")
    print(f"Hit@{args.top_k}: {hits / len(verified):.3f}")
    print(f"MRR@{args.top_k}: {sum(reciprocal_ranks) / len(verified):.3f}")
    print(f"Verified unsupported cases (not scored for retrieval): {unsupported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
