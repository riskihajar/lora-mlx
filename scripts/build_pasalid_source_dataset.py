#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from lora_mlx.paths import DEFAULT_DATA_DIR


DEFAULT_QA_BANK = DEFAULT_DATA_DIR / "pasalid" / "qa_bank_json_large.jsonl"
DEFAULT_SOURCE_DIR = DEFAULT_DATA_DIR / "pasalid_source"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_source_payload(answer_field: str) -> dict:
    payload = json.loads(answer_field)
    return {
        "source_type": str(payload.get("source_type", "")).strip(),
        "source_number": str(payload.get("source_number", "")).strip(),
        "source_year": str(payload.get("source_year", "")).strip(),
        "source_article": str(payload.get("source_article", "")).strip(),
    }


def build_source_example(row: dict) -> dict:
    source_payload = normalize_source_payload(row["answer"])
    target = json.dumps(source_payload, ensure_ascii=True, separators=(",", ":"))
    text = f"Q: {row['question']}\nA: {target}"
    return {
        "text": text,
        "law_id": row["law_id"],
        "article_number": row["article_number"],
    }


def split_by_law(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    import random
    from collections import defaultdict

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["law_id"]].append(row)

    law_ids = list(grouped.keys())
    random.Random(seed).shuffle(law_ids)
    if len(law_ids) < 4:
        raise ValueError("Need at least 4 laws to build source dataset splits")

    valid_count = max(1, len(law_ids) // 10)
    test_count = max(1, len(law_ids) // 10)
    train_laws = law_ids[: len(law_ids) - valid_count - test_count]
    valid_laws = law_ids[len(train_laws) : len(train_laws) + valid_count]
    test_laws = law_ids[len(train_laws) + valid_count :]

    return {
        "train": [row for law in train_laws for row in grouped[law]],
        "valid": [row for law in valid_laws for row in grouped[law]],
        "test": [row for law in test_laws for row in grouped[law]],
        "train_laws": train_laws,
        "valid_laws": valid_laws,
        "test_laws": test_laws,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps({"text": row["text"]}, ensure_ascii=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Pasal.id source-prediction dataset from QA bank.")
    parser.add_argument("--input", default=str(DEFAULT_QA_BANK), help="QA bank input JSONL")
    parser.add_argument("--output-dir", default=str(DEFAULT_SOURCE_DIR), help="Output directory for source dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_rows(Path(args.input))
    source_rows = [build_source_example(row) for row in rows]
    split = split_by_law(source_rows, args.seed)
    output_dir = Path(args.output_dir)

    write_jsonl(output_dir / "train.jsonl", split["train"])
    write_jsonl(output_dir / "valid.jsonl", split["valid"])
    write_jsonl(output_dir / "test.jsonl", split["test"])

    manifest = {
        "total_rows": len(source_rows),
        "train_rows": len(split["train"]),
        "valid_rows": len(split["valid"]),
        "test_rows": len(split["test"]),
        "train_laws": split["train_laws"],
        "valid_laws": split["valid_laws"],
        "test_laws": split["test_laws"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
