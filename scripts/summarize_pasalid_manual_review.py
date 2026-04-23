#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


def to_int(value: str) -> int:
    value = value.strip()
    return int(value) if value else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Pasal.id manual review scores from a CSV file.")
    parser.add_argument("--input", required=True, help="CSV file containing manual review rows")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = Path(args.input)
    with path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError("No manual review rows found")

    total = len(rows)
    factual_scores = [to_int(row.get("Factual Correctness (0/1/2)", "0")) for row in rows]
    evidence_scores = [to_int(row.get("Evidence Support (0/1/2)", "0")) for row in rows]
    source_scores = [to_int(row.get("Source Traceability (0/1/2)", "0")) for row in rows]

    result = {
        "rows": total,
        "factual_correctness_avg": sum(factual_scores) / total,
        "evidence_support_avg": sum(evidence_scores) / total,
        "source_traceability_avg": sum(source_scores) / total,
        "evidence_support_rate": sum(1 for score in evidence_scores if score >= 1) / total,
        "unsupported_answer_rate": sum(1 for score in evidence_scores if score == 0) / total,
        "factual_nonzero_rate": sum(1 for score in factual_scores if score >= 1) / total,
        "source_traceability_rate": sum(1 for score in source_scores if score >= 1) / total,
    }

    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
