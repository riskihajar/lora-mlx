#!/usr/bin/env python3

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from lora_mlx.paths import DEFAULT_DATA_DIR


DEFAULT_QA_BANK = DEFAULT_DATA_DIR / "pasalid" / "qa_bank.jsonl"
DEFAULT_SPLIT_DIR = DEFAULT_DATA_DIR / "pasalid"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict], include_source_doc: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            text = f"Q: {row['question']}\nA: {row['answer']}"
            if include_source_doc:
                text = (
                    f"Dokumen sumber: {row['source_doc']}\n"
                    f"Referensi: {row['source_reference']}\n"
                    f"Q: {row['question']}\n"
                    f"A: {row['answer']}"
                )
            handle.write(json.dumps({"text": text}, ensure_ascii=True) + "\n")


def split_rows(rows: list[dict], seed: int, valid_law_count: int | None = None, test_law_count: int | None = None):
    by_law = defaultdict(list)
    for row in rows:
        by_law[row["law_id"]].append(row)

    law_ids = list(by_law.keys())
    rng = random.Random(seed)
    rng.shuffle(law_ids)

    if len(law_ids) < 4:
        raise ValueError("Need at least 4 laws to build train/valid/test_seen/test_unseen splits.")

    valid_count = valid_law_count if valid_law_count is not None else max(1, len(law_ids) // 10)
    test_count = test_law_count if test_law_count is not None else max(1, len(law_ids) // 10)
    if valid_count < 1 or test_count < 1:
        raise ValueError("valid and test law counts must be at least 1")
    if valid_count + test_count >= len(law_ids):
        raise ValueError("valid and test law counts must leave at least one train law")
    train_laws = law_ids[: len(law_ids) - valid_count - test_count]
    valid_laws = law_ids[len(train_laws) : len(train_laws) + valid_count]
    test_unseen_laws = law_ids[len(train_laws) + valid_count :]

    train_rows = []
    test_seen_rows = []
    for law_id in train_laws:
        law_rows = by_law[law_id][:]
        rng.shuffle(law_rows)
        split_idx = max(1, int(len(law_rows) * 0.7))
        if split_idx >= len(law_rows):
            split_idx = len(law_rows) - 1
        train_rows.extend(law_rows[:split_idx])
        test_seen_rows.extend(law_rows[split_idx:])

    valid_rows = [row for law_id in valid_laws for row in by_law[law_id]]
    test_unseen_rows = [row for law_id in test_unseen_laws for row in by_law[law_id]]

    return {
        "train": train_rows,
        "valid": valid_rows,
        "test_seen": test_seen_rows,
        "test_unseen": test_unseen_rows,
        "train_laws": train_laws,
        "valid_laws": valid_laws,
        "test_unseen_laws": test_unseen_laws,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split Pasal.id QA bank into train/valid/test_seen/test_unseen.")
    parser.add_argument("--input", default=str(DEFAULT_QA_BANK), help="Input QA bank JSONL")
    parser.add_argument("--output-dir", default=str(DEFAULT_SPLIT_DIR), help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-laws", type=int, default=4, help="Minimum distinct laws required to build full experiment splits")
    parser.add_argument("--valid-law-count", type=int, default=None, help="Optional number of validation laws")
    parser.add_argument("--test-law-count", type=int, default=None, help="Optional number of held-out test laws")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_rows(Path(args.input))
    law_count = len({row["law_id"] for row in rows})
    if law_count < args.min_laws:
        raise ValueError(
            f"Need at least {args.min_laws} laws to build experiment splits, but only found {law_count}."
        )
    splits = split_rows(rows, args.seed, valid_law_count=args.valid_law_count, test_law_count=args.test_law_count)
    output_dir = Path(args.output_dir)

    write_jsonl(output_dir / "train.jsonl", splits["train"], include_source_doc=True)
    write_jsonl(output_dir / "valid.jsonl", splits["valid"], include_source_doc=True)
    write_jsonl(output_dir / "test_seen.jsonl", splits["test_seen"], include_source_doc=False)
    write_jsonl(output_dir / "test_unseen.jsonl", splits["test_unseen"], include_source_doc=False)
    write_jsonl(output_dir / "test_seen_with_context.jsonl", splits["test_seen"], include_source_doc=True)
    write_jsonl(output_dir / "test_unseen_with_context.jsonl", splits["test_unseen"], include_source_doc=True)

    manifest = {
        "total_rows": len(rows),
        "train_rows": len(splits["train"]),
        "valid_rows": len(splits["valid"]),
        "test_seen_rows": len(splits["test_seen"]),
        "test_unseen_rows": len(splits["test_unseen"]),
        "train_laws": splits["train_laws"],
        "valid_laws": splits["valid_laws"],
        "test_unseen_laws": splits["test_unseen_laws"],
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=True))


if __name__ == "__main__":
    main()
