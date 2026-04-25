#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path


SCORE_FIELDS = ["factual_correctness", "evidence_support", "source_traceability", "naturalness"]


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize(rows: list[dict]) -> dict:
    summary = {"rows": len(rows), "metrics": {}, "winner_counts": {}, "label_counts": {}}
    for field in SCORE_FIELDS:
        b_values = [row["B"][field] for row in rows]
        d_values = [row["D"][field] for row in rows]
        summary["metrics"][field] = {
            "B_avg": average(b_values),
            "D_avg": average(d_values),
            "D_minus_B": average(d_values) - average(b_values),
        }
        summary["winner_counts"][field] = dict(Counter(row["winners"][field] for row in rows))
    summary["winner_counts"]["overall"] = dict(Counter(row["winners"]["overall"] for row in rows))
    summary["label_counts"]["B"] = dict(Counter(label for row in rows for label in row.get("B_error_labels", [])))
    summary["label_counts"]["D"] = dict(Counter(label for row in rows for label in row.get("D_error_labels", [])))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize pairwise Pasal.id B vs D review results.")
    parser.add_argument("--input", required=True, help="Pairwise review JSONL file")
    parser.add_argument("--output", help="Optional summary JSON output path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize(load_rows(Path(args.input)))
    text = json.dumps(summary, ensure_ascii=True, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
