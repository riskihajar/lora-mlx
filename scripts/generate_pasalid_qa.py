#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

from lora_mlx.openai_compat import generate_text, validate_client
from lora_mlx.paths import DEFAULT_DATA_DIR


DEFAULT_DOC_UNITS = DEFAULT_DATA_DIR / "pasalid" / "doc_units.jsonl"
DEFAULT_QA_BANK = DEFAULT_DATA_DIR / "pasalid" / "qa_bank.jsonl"


SYSTEM_PROMPT = """You generate Indonesian legal QA pairs from source-backed document units.

Rules:
- Output strict JSON only.
- The JSON must be an object with one key named items.
- items must be a JSON array.
- Each item in the array must contain exactly these keys: question, answer, question_type, difficulty.
- The question must be answerable from the source document unit.
- The answer must be concise, factually grounded, and must use a structured two-line format.
- Prefer short declarative answers rather than bullet lists or numbered lists.
- Do not use multiple-choice formatting.
- The answer must have exactly two lines:
  1. "Jawaban: <concise factual answer>"
  2. "Sumber: <reference>."
- The source reference must exactly match the provided source_reference field.
- Never omit the source sentence.
- Never rewrite or paraphrase the source reference.
- Never add extra lines before or after the two required lines.
- Do not invent legal facts not present in the source.
- Use Indonesian.
"""


def load_doc_units(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_user_prompt(unit: dict) -> str:
    return json.dumps(
        {
            "title": unit["title"],
            "short_title": unit["short_title"],
            "source_reference": unit["source_reference"],
            "question_type_candidate": unit["question_type_candidate"],
            "source_doc": unit["source_doc"],
            "instruction": "Generate multiple high-quality legal QA pairs grounded in the source.",
            "requirements": {
                "items_count": 3,
                "question_diversity": [
                    "one direct article question",
                    "one substantive legal question",
                    "one paraphrased or source-traceability question",
                ],
                "answer_style": [
                    "exactly two lines",
                    "line 1 must start with Jawaban:",
                    "line 2 must start with Sumber:",
                    "no numbering or bullet points",
                    "must end with an exact source sentence",
                    "source sentence must use the exact provided source_reference",
                ],
            },
        },
        ensure_ascii=True,
        indent=2,
    )


def parse_generation(text: str) -> list[dict]:
    payload = json.loads(text)
    items = payload.get("items", [])
    if not isinstance(items, list) or not items:
        raise ValueError("Generation output must contain a non-empty items array")
    return items


def load_existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {(row["law_id"], row["article_number"]) for row in rows}


def prompt_input(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Pasal.id QA bank from canonical document units.")
    parser.add_argument("--input", default=str(DEFAULT_DOC_UNITS), help="Canonical doc units JSONL")
    parser.add_argument("--output", default=str(DEFAULT_QA_BANK), help="Output QA bank JSONL")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of doc units to process")
    parser.add_argument("--offset", type=int, default=0, help="Starting offset in the doc unit file")
    parser.add_argument("--append", action="store_true", help="Append to output instead of overwriting")
    parser.add_argument("--interactive", action="store_true", help="Prompt for missing choices interactively")
    parser.add_argument("--validate-client", action="store_true", help="Validate configured OpenAI-like client before generation")
    parser.add_argument("--questions-per-unit", type=int, default=3, help="Target number of questions per document unit")
    parser.add_argument("--retries", type=int, default=3, help="Retries per document unit on generation failure")
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="Optional delay between successful generations")
    parser.add_argument("--skip-existing", action="store_true", help="Skip document units that already exist in the output file")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    limit = args.limit
    offset = args.offset
    append = args.append
    questions_per_unit = args.questions_per_unit
    retries = args.retries
    sleep_seconds = args.sleep_seconds
    skip_existing = args.skip_existing

    if args.interactive:
        input_path = Path(prompt_input("Input doc units", str(input_path)))
        output_path = Path(prompt_input("Output QA bank", str(output_path)))
        limit = int(prompt_input("Max doc units to process", str(limit)))
        offset = int(prompt_input("Offset", str(offset)))
        questions_per_unit = int(prompt_input("Questions per doc unit", str(questions_per_unit)))
        append = prompt_input("Append output? (yes/no)", "no").lower() in {"y", "yes"}
        skip_existing = prompt_input("Skip existing units? (yes/no)", "yes").lower() in {"y", "yes"}

    if args.validate_client:
        result = validate_client()
        print(json.dumps(result, ensure_ascii=True))

    units = load_doc_units(input_path)
    selected = units[offset : offset + limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    existing_keys = load_existing_keys(output_path) if skip_existing and append else set()

    generated = 0
    processed = 0
    skipped = 0
    with output_path.open(mode) as handle:
        for unit in selected:
            unit_key = (unit["law_id"], str(unit["article_number"]))
            if unit_key in existing_keys:
                skipped += 1
                print(f"skip law={unit['law_id']} article={unit['article_number']}")
                continue

            prompt_payload = json.loads(build_user_prompt(unit))
            prompt_payload["requirements"]["items_count"] = questions_per_unit
            items = None
            last_error = None
            for attempt in range(1, retries + 1):
                try:
                    raw = generate_text(SYSTEM_PROMPT, json.dumps(prompt_payload, ensure_ascii=True, indent=2))
                    items = parse_generation(raw)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    print(
                        f"retry {attempt}/{retries} law={unit['law_id']} article={unit['article_number']} error={exc}"
                    )
                    time.sleep(1.0)
            if items is None:
                raise RuntimeError(
                    f"Failed to generate QA for law={unit['law_id']} article={unit['article_number']}"
                ) from last_error

            for qa in items[:questions_per_unit]:
                row = {
                    "law_id": unit["law_id"],
                    "frbr_uri": unit["frbr_uri"],
                    "title": unit["title"],
                    "short_title": unit["short_title"],
                    "article_number": unit["article_number"],
                    "source_reference": unit["source_reference"],
                    "source_doc": unit["source_doc"],
                    "question_type_candidate": unit["question_type_candidate"],
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "question_type": qa["question_type"],
                    "difficulty": qa["difficulty"],
                }
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                generated += 1
            processed += 1
            existing_keys.add(unit_key)
            print(
                f"done {processed} law={unit['law_id']} article={unit['article_number']} generated_total={generated}"
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    print(f"generated={generated}")
    print(f"processed_units={processed}")
    print(f"skipped_units={skipped}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
