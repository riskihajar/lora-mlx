#!/usr/bin/env python3

import argparse
import json
import re
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


def parse_answer_payload(answer_field: str) -> dict:
    return json.loads(answer_field)


def build_source_example(row: dict) -> dict:
    source_payload = normalize_source_payload(row["answer"])
    target = json.dumps(source_payload, ensure_ascii=True, separators=(",", ":"))
    text = f"Q: {row['question']}\nA: {target}"
    return {
        "text": text,
        "law_id": row["law_id"],
        "article_number": row["article_number"],
        "is_source_explicit": question_mentions_source(row["question"], source_payload),
        "source_question_variant": "observed",
    }


def build_implicit_source_example(row: dict) -> dict | None:
    answer_payload = parse_answer_payload(row["answer"])
    source_payload = normalize_source_payload(row["answer"])
    answer_text = sanitize_implicit_clue(str(answer_payload.get("answer", "")).strip(), source_payload)
    if not answer_text:
        return None
    question = f"Rujukan hukum mana yang mengatur ketentuan berikut: {answer_text}?"
    if question_mentions_source(question, source_payload):
        return None
    target = json.dumps(source_payload, ensure_ascii=True, separators=(",", ":"))
    return {
        "text": f"Q: {question}\nA: {target}",
        "law_id": row["law_id"],
        "article_number": row["article_number"],
        "is_source_explicit": False,
        "source_question_variant": "generated_implicit",
    }


def sanitize_implicit_clue(text: str, source_payload: dict) -> str:
    cleaned = text
    source_number = source_payload.get("source_number", "")
    source_year = source_payload.get("source_year", "")
    source_article = source_payload.get("source_article", "")

    if source_article:
        cleaned = re.sub(rf"\bpasal\s*{re.escape(source_article)}\b", "ketentuan ini", cleaned, flags=re.IGNORECASE)
    if source_number and source_year:
        cleaned = re.sub(
            rf"\b(?:uu|undang-undang)\s*(?:nomor|no\.)?\s*{re.escape(source_number)}\s*(?:tahun)?\s*{re.escape(source_year)}\b",
            "undang-undang terkait",
            cleaned,
            flags=re.IGNORECASE,
        )
    if source_year:
        cleaned = re.sub(rf"\b{re.escape(source_year)}\b", "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip(" .?")


def question_mentions_source(question: str, source_payload: dict) -> bool:
    lowered = question.lower()
    checks = []
    source_number = source_payload.get("source_number", "")
    source_year = source_payload.get("source_year", "")
    source_article = source_payload.get("source_article", "")

    if source_number:
        checks.append(rf"(?:uu|undang-undang)\s*(?:nomor|no\.)?\s*{re.escape(source_number.lower())}\b")
    if source_year:
        checks.append(rf"\b{re.escape(source_year.lower())}\b")
    if source_article:
        checks.append(rf"pasal\s*{re.escape(source_article.lower())}\b")

    return any(re.search(pattern, lowered) for pattern in checks)


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

    rng = random.Random(seed)
    train_rows = []
    test_seen_rows = []
    for law in train_laws:
        law_rows = grouped[law][:]
        rng.shuffle(law_rows)
        split_idx = max(1, int(len(law_rows) * 0.7))
        if split_idx >= len(law_rows):
            split_idx = len(law_rows) - 1
        train_rows.extend(law_rows[:split_idx])
        test_seen_rows.extend(law_rows[split_idx:])

    test_unseen_rows = [row for law in test_laws for row in grouped[law]]

    return {
        "train": train_rows,
        "valid": [row for law in valid_laws for row in grouped[law]],
        "test": test_unseen_rows,
        "test_seen": test_seen_rows,
        "test_unseen": test_unseen_rows,
        "train_laws": train_laws,
        "valid_laws": valid_laws,
        "test_laws": test_laws,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps({"text": row["text"]}, ensure_ascii=True) + "\n")


def write_split(output_dir: Path, split: dict[str, list[dict]], manifest: dict) -> None:
    write_jsonl(output_dir / "train.jsonl", split["train"])
    write_jsonl(output_dir / "valid.jsonl", split["valid"])
    write_jsonl(output_dir / "test.jsonl", split["test"])
    write_jsonl(output_dir / "test_seen.jsonl", split.get("test_seen", []))
    write_jsonl(output_dir / "test_unseen.jsonl", split.get("test_unseen", split["test"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n")


def filter_split(split: dict[str, list[dict]], *, explicit: bool) -> dict[str, list[dict]]:
    return {
        "train": [row for row in split["train"] if row["is_source_explicit"] is explicit],
        "valid": [row for row in split["valid"] if row["is_source_explicit"] is explicit],
        "test": [row for row in split["test"] if row["is_source_explicit"] is explicit],
        "test_seen": [row for row in split.get("test_seen", []) if row["is_source_explicit"] is explicit],
        "test_unseen": [row for row in split.get("test_unseen", split["test"]) if row["is_source_explicit"] is explicit],
        "train_laws": split["train_laws"],
        "valid_laws": split["valid_laws"],
        "test_laws": split["test_laws"],
    }


def build_manifest(split: dict, rows: list[dict], variant: str) -> dict:
    explicit_rows = sum(1 for row in rows if row["is_source_explicit"])
    implicit_rows = len(rows) - explicit_rows
    generated_implicit_rows = sum(1 for row in rows if row.get("source_question_variant") == "generated_implicit")
    return {
        "variant": variant,
        "total_rows": len(rows),
        "explicit_rows": explicit_rows,
        "implicit_rows": implicit_rows,
        "generated_implicit_rows": generated_implicit_rows,
        "train_rows": len(split["train"]),
        "valid_rows": len(split["valid"]),
        "test_rows": len(split["test"]),
        "test_seen_rows": len(split.get("test_seen", [])),
        "test_unseen_rows": len(split.get("test_unseen", split["test"])),
        "train_laws": split.get("train_laws", []),
        "valid_laws": split.get("valid_laws", []),
        "test_laws": split.get("test_laws", []),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Pasal.id source-prediction dataset from QA bank.")
    parser.add_argument("--input", default=str(DEFAULT_QA_BANK), help="QA bank input JSONL")
    parser.add_argument("--output-dir", default=str(DEFAULT_SOURCE_DIR), help="Output directory for source dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_rows(Path(args.input))
    observed_rows = [build_source_example(row) for row in rows]
    generated_implicit_rows = [example for row in rows if (example := build_implicit_source_example(row)) is not None]
    source_rows = observed_rows
    split = split_by_law(source_rows, args.seed)
    implicit_augmented_split = split_by_law(generated_implicit_rows, args.seed)
    output_dir = Path(args.output_dir)

    manifest = build_manifest(split, source_rows, "all")
    write_split(output_dir, split, manifest)

    explicit_split = filter_split(split, explicit=True)
    implicit_split = filter_split(split, explicit=False)
    implicit_augmented_rows = [row for row in source_rows if not row["is_source_explicit"]] + generated_implicit_rows
    implicit_augmented_split = {
        "train": implicit_split["train"] + implicit_augmented_split["train"],
        "valid": implicit_split["valid"] + implicit_augmented_split["valid"],
        "test": implicit_split["test"] + implicit_augmented_split["test"],
        "test_seen": implicit_split["test_seen"] + implicit_augmented_split["test_seen"],
        "test_unseen": implicit_split["test_unseen"] + implicit_augmented_split["test_unseen"],
        "train_laws": split["train_laws"],
        "valid_laws": split["valid_laws"],
        "test_laws": split["test_laws"],
    }
    explicit_manifest = build_manifest(explicit_split, [row for row in source_rows if row["is_source_explicit"]], "explicit")
    implicit_manifest = build_manifest(implicit_augmented_split, implicit_augmented_rows, "implicit")
    write_split(output_dir / "explicit", explicit_split, explicit_manifest)
    write_split(output_dir / "implicit", implicit_augmented_split, implicit_manifest)

    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
