"""Build a Doc-to-LoRA-friendly Pasal.id eval split.

Reshapes ``data/pasalid/qa_bank_json_native_expanded_clean.jsonl`` and the
existing seen/unseen split into the schema expected by the cloud eval entry
point: separate ``doc`` and ``question`` fields, deduplicated documents, and
an explicit doc -> qa_ids mapping so each document is internalized once.

Outputs:

    data/pasalid/d2l_eval/test_seen.jsonl
    data/pasalid/d2l_eval/test_unseen.jsonl
    data/pasalid/d2l_eval/docs_seen.jsonl
    data/pasalid/d2l_eval/docs_unseen.jsonl
    data/pasalid/d2l_eval/manifest.json

Usage:

    python3 cloud/build_pasalid_d2l_eval.py
    python3 cloud/build_pasalid_d2l_eval.py --subset 30

The optional ``--subset`` flag samples up to N rows per split (deterministic
seed) for cheap pipeline validation runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
QA_BANK = REPO_ROOT / "data/pasalid/qa_bank_json_native_expanded_clean.jsonl"
SPLIT_DIR = REPO_ROOT / "data/pasalid/json_native_expanded_clean_split"
OUTPUT_DIR = REPO_ROOT / "data/pasalid/d2l_eval"


def _doc_id(law_id: str, article_number: str, source_doc: str) -> str:
    """Stable id per (law, article, doc-content) triplet."""
    digest = hashlib.sha1(source_doc.encode("utf-8")).hexdigest()[:8]
    safe_law = law_id.replace("/akn/id/act/uu/", "uu_").replace("/", "_")
    safe_art = str(article_number).replace("/", "_")
    return f"{safe_law}_art{safe_art}_{digest}"


def _qa_id(doc_id: str, idx: int) -> str:
    return f"{doc_id}_q{idx}"


def _load_qa_bank() -> list[dict]:
    rows = []
    with QA_BANK.open() as fp:
        for line in fp:
            rows.append(json.loads(line))
    return rows


def _load_split_questions(name: str) -> list[str]:
    """Return the list of ``Q:`` strings present in a given split file.

    The split files store rows as ``{"text": "Q: ...\\nA: ..."}``. We keep
    duplicates because the same question may appear under multiple laws.
    """
    path = SPLIT_DIR / f"{name}.jsonl"
    questions: list[str] = []
    with path.open() as fp:
        for line in fp:
            row = json.loads(line)
            text = row["text"]
            head, _, _ = text.partition("\nA:")
            assert head.startswith("Q: "), f"unexpected row format: {head!r}"
            questions.append(head[len("Q: "):].strip())
    return questions


def _load_split_answers(name: str) -> list[str]:
    """Return the list of ``A:`` strings present in a given split file.

    Used to disambiguate rows when the bank has multiple records with the
    same question across different laws (template phrasing).
    """
    path = SPLIT_DIR / f"{name}.jsonl"
    answers: list[str] = []
    with path.open() as fp:
        for line in fp:
            row = json.loads(line)
            text = row["text"]
            _, sep, tail = text.partition("\nA:")
            assert sep, f"missing A: separator in {text!r}"
            answers.append(tail.strip())
    return answers


def _build_split(
    bank: list[dict],
    split_name: str,
    subset: int | None,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Return ``(qa_rows, doc_rows)`` for a split.

    Matches each split row against the canonical bank by ``(question, answer)``
    pair to handle template questions that recur across multiple laws.
    """
    pairs = list(
        zip(_load_split_questions(split_name), _load_split_answers(split_name))
    )

    bank_index: dict[tuple[str, str], list[dict]] = {}
    for r in bank:
        key = (r["question"].strip(), r["answer"].strip())
        bank_index.setdefault(key, []).append(r)

    matched: list[dict] = []
    used_ids: set[int] = set()
    misses = 0
    for q, a in pairs:
        cands = bank_index.get((q, a), [])
        chosen = None
        for cand in cands:
            if id(cand) not in used_ids:
                chosen = cand
                used_ids.add(id(cand))
                break
        if chosen is None:
            misses += 1
            continue
        matched.append(chosen)

    if misses:
        raise RuntimeError(
            f"{misses} split rows did not match bank entries by (Q, A) pair"
        )

    if subset is not None and len(matched) > subset:
        rnd = random.Random(seed)
        matched = rnd.sample(matched, subset)

    # Deterministic ordering for reproducibility.
    matched.sort(key=lambda r: (r["law_id"], str(r["article_number"]), r["question"]))

    docs: "OrderedDict[str, dict]" = OrderedDict()
    qa_rows: list[dict] = []
    per_doc_counter: dict[str, int] = {}

    for r in matched:
        doc_id = _doc_id(r["law_id"], r["article_number"], r["source_doc"])
        idx = per_doc_counter.get(doc_id, 0)
        per_doc_counter[doc_id] = idx + 1
        qa_id = _qa_id(doc_id, idx)

        qa_rows.append(
            {
                "qa_id": qa_id,
                "doc_id": doc_id,
                "law_id": r["law_id"],
                "article_number": r["article_number"],
                "source_reference": r["source_reference"],
                "question": r["question"],
                "expected_answer": r["answer"],
            }
        )
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "law_id": r["law_id"],
                "article_number": r["article_number"],
                "source_reference": r["source_reference"],
                "doc": r["source_doc"],
                "qa_ids": [qa_id],
            }
        else:
            docs[doc_id]["qa_ids"].append(qa_id)

    return qa_rows, list(docs.values())


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="sample up to N rows per split (default: full split)",
    )
    parser.add_argument("--seed", type=int, default=20260525)
    args = parser.parse_args()

    bank = _load_qa_bank()
    print(f"[build] qa_bank rows: {len(bank)}")

    splits = {}
    for split_name in ("test_seen", "test_unseen"):
        qa_rows, doc_rows = _build_split(
            bank, split_name, args.subset, args.seed + hash(split_name) % 10_000
        )
        qa_path = OUTPUT_DIR / f"{split_name}.jsonl"
        doc_path = OUTPUT_DIR / f"docs_{split_name.split('_', 1)[1]}.jsonl"
        n_qa = _write_jsonl(qa_path, qa_rows)
        n_doc = _write_jsonl(doc_path, doc_rows)
        print(
            f"[build] {split_name}: qa={n_qa} docs={n_doc} -> {qa_path.name},"
            f" {doc_path.name}"
        )
        splits[split_name] = {"qa": n_qa, "docs": n_doc}

    manifest = {
        "source_qa_bank": str(QA_BANK.relative_to(REPO_ROOT)),
        "source_split_dir": str(SPLIT_DIR.relative_to(REPO_ROOT)),
        "subset_per_split": args.subset,
        "seed": args.seed,
        "splits": splits,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[build] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
