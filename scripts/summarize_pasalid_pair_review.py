#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


SCORE_FIELDS = ["factual_correctness", "evidence_support", "source_traceability"]


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def compare_scores(left: list[dict], right: list[dict], left_label: str, right_label: str) -> dict:
    if len(left) != len(right):
        raise ValueError(f"Review row counts do not match: {left_label}={len(left)} {right_label}={len(right)}")

    summary = {"rows": len(left), "labels": [left_label, right_label], "metrics": {}}
    for field in SCORE_FIELDS:
        left_values = [row[field] for row in left]
        right_values = [row[field] for row in right]
        left_wins = sum(1 for lval, rval in zip(left_values, right_values) if lval > rval)
        right_wins = sum(1 for lval, rval in zip(left_values, right_values) if rval > lval)
        ties = len(left_values) - left_wins - right_wins
        summary["metrics"][field] = {
            f"{left_label}_avg": average(left_values),
            f"{right_label}_avg": average(right_values),
            f"{right_label}_minus_{left_label}": average(right_values) - average(left_values),
            f"{left_label}_wins": left_wins,
            f"{right_label}_wins": right_wins,
            "ties": ties,
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize paired Pasal.id LLM review files.")
    parser.add_argument("--left", required=True, help="Left review JSONL file")
    parser.add_argument("--right", required=True, help="Right review JSONL file")
    parser.add_argument("--left-label", default="B", help="Label for left file")
    parser.add_argument("--right-label", default="D", help="Label for right file")
    parser.add_argument("--output", help="Optional summary JSON output path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = compare_scores(load_rows(Path(args.left)), load_rows(Path(args.right)), args.left_label, args.right_label)
    text = json.dumps(summary, ensure_ascii=True, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
