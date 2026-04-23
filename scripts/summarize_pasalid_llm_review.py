#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize GPT-assisted Pasal.id review results.")
    parser.add_argument("--input", required=True, help="Reviewed JSONL file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_rows(Path(args.input))
    if not rows:
        raise ValueError("No review rows found")

    total = len(rows)
    label_counter = Counter()
    factual_scores = []
    evidence_scores = []
    source_scores = []
    confidence_scores = []

    for row in rows:
        factual_scores.append(row["factual_correctness"])
        evidence_scores.append(row["evidence_support"])
        source_scores.append(row["source_traceability"])
        confidence_scores.append(row["confidence"])
        label_counter.update(row["error_labels"])

    summary = {
        "rows": total,
        "factual_correctness_avg": sum(factual_scores) / total,
        "evidence_support_avg": sum(evidence_scores) / total,
        "source_traceability_avg": sum(source_scores) / total,
        "evidence_support_rate": sum(1 for score in evidence_scores if score >= 1) / total,
        "unsupported_answer_rate": label_counter["unsupported-answer"] / total,
        "source_missing_rate": label_counter["source-missing"] / total,
        "source_wrong_rate": label_counter["source-wrong"] / total,
        "factually_wrong_rate": label_counter["factually-wrong"] / total,
        "avg_confidence": sum(confidence_scores) / total,
        "label_counts": dict(label_counter),
    }

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
