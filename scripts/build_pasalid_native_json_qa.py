#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

from lora_mlx.paths import DEFAULT_DATA_DIR


DEFAULT_DOC_UNITS = DEFAULT_DATA_DIR / "pasalid" / "doc_units.jsonl"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "pasalid" / "qa_bank_json_native_expanded.jsonl"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def first_sentence(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", normalize_space(text))
    return sentences[0].strip() if sentences else normalize_space(text)


def short_topic(text: str, max_words: int = 14) -> str:
    words = re.findall(r"[A-Za-z0-9]+(?:/[A-Za-z0-9]+)?", normalize_space(text))
    return " ".join(words[:max_words]).strip(" ,.-")


def is_usable(row: dict) -> bool:
    source_doc = normalize_space(str(row.get("source_doc", "")))
    article = str(row.get("article_number", "")).strip()
    if len(source_doc) < 60 or not article:
        return False
    lowered = source_doc.lower()
    noisy_markers = ["line truncated", "republik indonesia", " . ."]
    return not any(marker in lowered for marker in noisy_markers)


def source_payload(row: dict, answer: str) -> str:
    payload = {
        "answer": normalize_space(answer),
        "source_type": str(row.get("type", "UU")),
        "source_number": str(row.get("number", "")),
        "source_year": str(row.get("year", "")),
        "source_article": str(row.get("article_number", "")).strip(),
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def base_row(row: dict, question: str, answer: str, question_type: str, difficulty: str) -> dict:
    return {
        "law_id": row["law_id"],
        "frbr_uri": row.get("frbr_uri", row["law_id"]),
        "title": row.get("title", ""),
        "short_title": row.get("short_title", ""),
        "article_number": str(row.get("article_number", "")).strip(),
        "source_reference": row.get("source_reference", ""),
        "source_doc": normalize_space(row.get("source_doc", "")),
        "question_type_candidate": row.get("question_type_candidate", "article_qa"),
        "question": question,
        "answer": source_payload(row, answer),
        "question_type": question_type,
        "difficulty": difficulty,
        "source_generation": "native_doc_unit_template",
    }


def build_rows_for_doc(row: dict) -> list[dict]:
    article = str(row.get("article_number", "")).strip()
    source_doc = normalize_space(row.get("source_doc", ""))
    law_label = f"{row.get('type', 'UU')} No. {row.get('number', '')} Tahun {row.get('year', '')}"
    sentence = first_sentence(source_doc)
    topic = short_topic(sentence)

    rows = [
        base_row(
            row,
            f"Apa ketentuan utama dalam Pasal {article} {law_label}?",
            f"Pasal {article} mengatur: {source_doc}",
            "direct_article_question",
            "easy",
        ),
        base_row(
            row,
            f"Ketentuan hukum apa yang relevan dengan frasa: {topic}?",
            sentence,
            "substantive_legal_question",
            "medium",
        ),
        base_row(
            row,
            f"Sumber hukum mana yang memuat ketentuan tentang {topic}?",
            sentence,
            "source_traceability_question",
            "medium",
        ),
    ]

    lowered = source_doc.lower()
    if "pidana" in lowered or "denda" in lowered:
        rows.append(
            base_row(
                row,
                f"Apa konsekuensi pidana atau denda yang diatur terkait {topic}?",
                source_doc,
                "sanction_question",
                "medium",
            )
        )
    elif re.search(r"\b(setiap orang|orang|pihak|pejabat|menteri|pemerintah)\b", lowered):
        rows.append(
            base_row(
                row,
                f"Siapa pihak atau subjek yang disebut dalam ketentuan tentang {topic}?",
                sentence,
                "subject_question",
                "medium",
            )
        )

    return rows


def deduplicate(rows: list[dict]) -> list[dict]:
    seen = set()
    unique_rows = []
    for row in rows:
        key = (row["law_id"], row["article_number"], row["question"], row["answer"])
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build native JSON answer+source QA from Pasal.id doc units.")
    parser.add_argument("--input", default=str(DEFAULT_DOC_UNITS), help="Input doc_units JSONL")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output QA bank JSONL")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    docs = [row for row in load_rows(input_path) if is_usable(row)]
    rows = deduplicate([qa_row for doc in docs for qa_row in build_rows_for_doc(doc)])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "doc_units": len(docs),
        "rows": len(rows),
        "laws": len({row["law_id"] for row in rows}),
        "question_types": sorted({row["question_type"] for row in rows}),
    }
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
