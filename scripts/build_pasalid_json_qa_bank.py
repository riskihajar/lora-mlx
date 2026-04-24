#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

from lora_mlx.paths import DEFAULT_DATA_DIR


DEFAULT_INPUT = DEFAULT_DATA_DIR / "pasalid" / "qa_bank_full.jsonl"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "pasalid" / "qa_bank_json_final.jsonl"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_law_parts(row: dict) -> tuple[str, str, str]:
    law_id = str(row.get("law_id", ""))
    match = re.search(r"/act/([^/]+)/(\d{4})/([^/]+)$", law_id)
    if match:
        law_type, year, number = match.groups()
        return law_type.upper(), number, year

    reference = str(row.get("source_reference", ""))
    ref_match = re.search(r"\b(UU|Undang-Undang)\b\s*(?:No\.?|Nomor)?\s*([A-Za-z0-9]+)\s*Tahun\s*(\d{4})", reference, flags=re.IGNORECASE)
    if ref_match:
        return "UU", ref_match.group(2), ref_match.group(3)

    raise ValueError(f"Cannot parse source parts for law_id={law_id!r}")


def clean_answer(answer: str) -> str:
    cleaned = re.sub(r"\s*\(\s*Sumber\s*:\s*.*\)\s*$", "", answer, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\[\s*Sumber\s*:\s*[^\]]*\]\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*Sumber\s*:\s*UU\s+No\.?\s+[^.]+\.\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\(\s*UU\s+No\.?\s+.*\)\s*\.?\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*Jawaban\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def convert_row(row: dict) -> dict:
    source_type, source_number, source_year = parse_law_parts(row)
    payload = {
        "answer": clean_answer(str(row["answer"])),
        "source_type": source_type,
        "source_number": source_number,
        "source_year": source_year,
        "source_article": str(row["article_number"]).strip(),
    }
    converted = dict(row)
    converted["answer"] = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    converted["answer_format"] = "json_answer_source"
    converted["source_conversion"] = "from_narrative_answer"
    return converted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Pasal.id QA bank answers to JSON answer+source format.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input QA bank JSONL")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output QA bank JSONL")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = [convert_row(row) for row in load_rows(input_path)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(rows),
        "laws": len({row["law_id"] for row in rows}),
    }
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
