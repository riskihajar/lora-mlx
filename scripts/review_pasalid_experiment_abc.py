#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from lora_mlx.eval_legal import (
    citation_component_score,
    citation_exact_match,
    exact_match,
    split_answer_and_source,
    token_f1,
)
from lora_mlx.paths import DEFAULT_PREDICTIONS_DIR


DEFAULT_EXPORT_DIR = DEFAULT_PREDICTIONS_DIR / "pasalid_experiment"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_metrics(rows: list[dict]) -> dict[str, float]:
    answer_ems = []
    answer_f1s = []
    citation_ems = []
    citation_scores = []
    for row in rows:
        pred_answer, pred_source = split_answer_and_source(row["prediction"])
        gold_answer, gold_source = split_answer_and_source(row["gold"])
        answer_ems.append(exact_match(pred_answer, gold_answer))
        answer_f1s.append(token_f1(pred_answer, gold_answer))
        citation_ems.append(citation_exact_match(pred_source, gold_source))
        citation_scores.append(citation_component_score(pred_source, gold_source))
    return {
        "answer_em": sum(answer_ems) / len(answer_ems),
        "answer_f1": sum(answer_f1s) / len(answer_f1s),
        "citation_em": sum(citation_ems) / len(citation_ems),
        "citation_component_score": sum(citation_scores) / len(citation_scores),
    }


def per_row_metrics(row: dict) -> dict[str, float]:
    pred_answer, pred_source = split_answer_and_source(row["prediction"])
    gold_answer, gold_source = split_answer_and_source(row["gold"])
    return {
        "answer_em": exact_match(pred_answer, gold_answer),
        "answer_f1": token_f1(pred_answer, gold_answer),
        "citation_em": citation_exact_match(pred_source, gold_source),
        "citation_component_score": citation_component_score(pred_source, gold_source),
    }


def print_block(title: str, content: str) -> None:
    print(f"{title}:")
    print(content.strip() if content.strip() else "(empty)")
    print()


def heuristic_flags(prediction: str, gold: str) -> list[str]:
    flags = []
    pred_answer, pred_source = split_answer_and_source(prediction)
    gold_answer, gold_source = split_answer_and_source(gold)

    if token_f1(pred_answer, gold_answer) < 0.15:
        flags.append("low_answer_overlap")
    if pred_source == "":
        flags.append("source_missing")
    if pred_source and citation_component_score(pred_source, gold_source) == 0.0:
        flags.append("source_wrong")
    absurd_markers = [
        "adalah bahaya",
        "satu-satunya kabupaten yang berada di wilayah kota",
        "undang-undang ini berlaku setiap tahun",
    ]
    lowered = pred_answer.lower()
    if any(marker in lowered for marker in absurd_markers):
        flags.append("possibly_factually_wrong")
    return flags


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review Pasal.id A/B/C outputs side by side.")
    parser.add_argument("--preset", required=True, help="Model preset prefix used in export filenames")
    parser.add_argument("--split", choices=["seen", "unseen"], default="seen", help="Which split to review")
    parser.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR), help="Directory containing exported A/B/C prediction files")
    parser.add_argument("--offset", type=int, default=0, help="Starting row offset")
    parser.add_argument("--limit", type=int, default=5, help="Number of rows to display")
    parser.add_argument("--summary-only", action="store_true", help="Only print aggregate metrics")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_dir = Path(args.export_dir)

    paths = {
        "A": export_dir / f"{args.preset}_{args.split}_A_base_no_context.jsonl",
        "B": export_dir / f"{args.preset}_{args.split}_B_base_with_context.jsonl",
        "C": export_dir / f"{args.preset}_{args.split}_C_adapter_no_context.jsonl",
    }

    rows = {label: load_rows(path) for label, path in paths.items()}

    summary = {label: summarize_metrics(label_rows) for label, label_rows in rows.items()}

    print("Summary Metrics")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print()

    if args.summary_only:
        return

    start = args.offset
    end = start + args.limit
    for idx in range(start, min(end, len(rows["A"]))):
        row_a = rows["A"][idx]
        row_b = rows["B"][idx]
        row_c = rows["C"][idx]
        print(f"=== Sample {idx + 1} ===")
        print_block("Question", row_a["prompt"])
        print_block("Gold", row_a["gold"])
        print_block("A - Base no context", row_a["prediction"])
        print_block("B - Base with context", row_b["prediction"])
        print_block("C - Adapter no context", row_c["prediction"])
        sample_metrics = {
            "A": per_row_metrics(row_a),
            "B": per_row_metrics(row_b),
            "C": per_row_metrics(row_c),
        }
        sample_flags = {
            "A": heuristic_flags(row_a["prediction"], row_a["gold"]),
            "B": heuristic_flags(row_b["prediction"], row_b["gold"]),
            "C": heuristic_flags(row_c["prediction"], row_c["gold"]),
        }
        print("Per-sample metrics:")
        print(json.dumps(sample_metrics, ensure_ascii=True, indent=2))
        print("Heuristic flags:")
        print(json.dumps(sample_flags, ensure_ascii=True, indent=2))
        print()


if __name__ == "__main__":
    main()
