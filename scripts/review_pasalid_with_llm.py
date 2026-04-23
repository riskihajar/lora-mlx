#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

from lora_mlx.openai_compat import generate_text, validate_client


SYSTEM_PROMPT = """You are reviewing Indonesian legal QA outputs.

Return strict JSON only with this exact schema:
{
  "factual_correctness": 0|1|2,
  "evidence_support": 0|1|2,
  "source_traceability": 0|1|2,
  "error_labels": ["supported-correct"|"supported-partial"|"unsupported-answer"|"factually-wrong"|"source-missing"|"source-wrong"],
  "confidence": 0.0,
  "reason": "short explanation"
}

Rubric:
- factual_correctness: 0 = wrong/absurd, 1 = partially correct, 2 = substantively correct
- evidence_support: 0 = not supported by source, 1 = partly supported, 2 = clearly supported by source
- source_traceability: 0 = missing/wrong, 1 = partial, 2 = correct and traceable

Rules:
- Use only the given gold and prediction.
- Keep reason short and concrete.
- confidence must be between 0 and 1.
- If the prediction is absurd or contradicts the gold, use `factually-wrong`.
- If the prediction gives an answer without support, use `unsupported-answer`.
- If source is absent, use `source-missing`.
- If source exists but is wrong, use `source-wrong`.
"""


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_user_prompt(row: dict) -> str:
    payload = {
        "question": row.get("prompt", ""),
        "gold": row.get("gold", ""),
        "prediction": row.get("prediction", ""),
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def parse_review(text: str) -> dict:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    return {
        "factual_correctness": int(payload["factual_correctness"]),
        "evidence_support": int(payload["evidence_support"]),
        "source_traceability": int(payload["source_traceability"]),
        "error_labels": list(payload["error_labels"]),
        "confidence": float(payload["confidence"]),
        "reason": str(payload["reason"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GPT-assisted review on Pasal.id prediction exports.")
    parser.add_argument("--input", required=True, help="Prediction export JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL review file")
    parser.add_argument("--limit", type=int, default=20, help="Maximum rows to review")
    parser.add_argument("--offset", type=int, default=0, help="Row offset")
    parser.add_argument("--validate-client", action="store_true", help="Validate configured OpenAI-like client first")
    parser.add_argument("--sleep-seconds", type=float, default=0.05, help="Delay between review calls")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.validate_client:
        print(json.dumps(validate_client(), ensure_ascii=True))

    rows = load_rows(Path(args.input))[args.offset : args.offset + args.limit]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reviewed = 0
    with output_path.open("w") as handle:
        for row in rows:
            review = parse_review(generate_text(SYSTEM_PROMPT, build_user_prompt(row)))
            out = {
                "index": row.get("index"),
                "prompt": row.get("prompt", ""),
                "gold": row.get("gold", ""),
                "prediction": row.get("prediction", ""),
                **review,
            }
            handle.write(json.dumps(out, ensure_ascii=True) + "\n")
            reviewed += 1
            print(f"reviewed={reviewed}")
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    print(f"output={output_path}")


if __name__ == "__main__":
    main()
