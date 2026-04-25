#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

from lora_mlx.openai_compat import generate_text, validate_client


SYSTEM_PROMPT = """You are comparing two Indonesian legal QA outputs, B and D.

Return strict JSON only with this exact schema:
{
  "B": {
    "factual_correctness": 0|1|2,
    "evidence_support": 0|1|2,
    "source_traceability": 0|1|2,
    "naturalness": 0|1|2
  },
  "D": {
    "factual_correctness": 0|1|2,
    "evidence_support": 0|1|2,
    "source_traceability": 0|1|2,
    "naturalness": 0|1|2
  },
  "winners": {
    "factual_correctness": "B"|"D"|"tie",
    "evidence_support": "B"|"D"|"tie",
    "source_traceability": "B"|"D"|"tie",
    "naturalness": "B"|"D"|"tie",
    "overall": "B"|"D"|"tie"
  },
  "B_error_labels": ["supported-correct"|"supported-partial"|"unsupported-answer"|"factually-wrong"|"source-missing"|"source-wrong"|"too-extractive"|"unnatural"],
  "D_error_labels": ["supported-correct"|"supported-partial"|"unsupported-answer"|"factually-wrong"|"source-missing"|"source-wrong"|"too-extractive"|"unnatural"],
  "reason": "short concrete explanation"
}

Rubric:
- factual_correctness: 0 = wrong/absurd, 1 = partially correct, 2 = substantively correct.
- evidence_support: 0 = unsupported by source, 1 = partly supported, 2 = clearly supported.
- source_traceability: 0 = missing/wrong citation, 1 = partial citation, 2 = correct and traceable citation.
- naturalness: 0 = incoherent/awkward, 1 = understandable but rough/extractive, 2 = natural concise Indonesian.

Rules:
- Use only the given source prompt, gold answer, and predictions.
- Penalize copied source text under naturalness with `too-extractive` if it reads like raw document text.
- Penalize wrong or missing JSON citation under source_traceability.
- Keep reason short.
"""


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_user_prompt(b_row: dict, d_row: dict) -> str:
    payload = {
        "question_and_source_prompt": b_row.get("prompt", ""),
        "gold": b_row.get("gold", ""),
        "B_prediction": b_row.get("prediction", ""),
        "D_prediction": d_row.get("prediction", ""),
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def parse_review(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pairwise GPT-assisted review for Pasal.id B vs D outputs.")
    parser.add_argument("--b-input", required=True, help="B prediction export JSONL file")
    parser.add_argument("--d-input", required=True, help="D prediction export JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL review file")
    parser.add_argument("--limit", type=int, default=30, help="Maximum paired rows to review")
    parser.add_argument("--offset", type=int, default=0, help="Row offset")
    parser.add_argument("--validate-client", action="store_true", help="Validate configured OpenAI-like client first")
    parser.add_argument("--sleep-seconds", type=float, default=0.05, help="Delay between review calls")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.validate_client:
        print(json.dumps(validate_client(), ensure_ascii=True))

    b_rows = load_rows(Path(args.b_input))[args.offset : args.offset + args.limit]
    d_rows = load_rows(Path(args.d_input))[args.offset : args.offset + args.limit]
    if len(b_rows) != len(d_rows):
        raise ValueError(f"Row counts do not match: B={len(b_rows)} D={len(d_rows)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for index, (b_row, d_row) in enumerate(zip(b_rows, d_rows), start=1):
            review = parse_review(generate_text(SYSTEM_PROMPT, build_user_prompt(b_row, d_row)))
            out = {
                "index": b_row.get("index"),
                "prompt": b_row.get("prompt", ""),
                "gold": b_row.get("gold", ""),
                "B_prediction": b_row.get("prediction", ""),
                "D_prediction": d_row.get("prediction", ""),
                **review,
            }
            handle.write(json.dumps(out, ensure_ascii=True) + "\n")
            print(f"reviewed={index}")
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    print(f"output={output_path}")


if __name__ == "__main__":
    main()
