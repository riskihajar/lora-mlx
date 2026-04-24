#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

from lora_mlx.paths import DEFAULT_DATA_DIR


DEFAULT_DOC_UNITS = DEFAULT_DATA_DIR / "pasalid" / "doc_units.jsonl"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "pasalid" / "realcase_eval.jsonl"
DEFAULT_SPLIT_DIR = DEFAULT_DATA_DIR / "pasalid" / "realcase_eval_split"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_usable(row: dict, max_source_chars: int) -> bool:
    source_doc = normalize_space(str(row.get("source_doc", "")))
    article = str(row.get("article_number", "")).strip()
    if len(source_doc) < 50 or not article:
        return False
    if max_source_chars > 0 and len(source_doc) > max_source_chars:
        return False
    lowered = source_doc.lower()
    noisy_markers = ["line truncated", " . .", "repu].ik", "repuellik", "repuelik"]
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


def base_row(row: dict, question: str, answer: str, question_type: str) -> dict:
    return {
        "law_id": row["law_id"],
        "frbr_uri": row.get("frbr_uri", row["law_id"]),
        "title": row.get("title", ""),
        "short_title": row.get("short_title", ""),
        "article_number": str(row.get("article_number", "")).strip(),
        "source_reference": row.get("source_reference", ""),
        "source_doc": normalize_space(row.get("source_doc", "")),
        "question": normalize_space(question),
        "answer": source_payload(row, answer),
        "question_type": question_type,
        "difficulty": "realcase_eval",
        "source_generation": "realcase_heuristic_from_doc_unit",
    }


def title_subject(row: dict) -> str:
    short_title = normalize_space(str(row.get("short_title", ""))).strip(" .")
    title = normalize_space(str(row.get("title", ""))).strip(" .")
    subject = short_title or title
    subject = re.sub(r"^Kabupaten\s+", "Kabupaten ", subject, flags=re.IGNORECASE)
    return subject[:120].strip()


def extract_region(row: dict) -> str:
    subject = title_subject(row)
    match = re.search(r"(Kabupaten|Kota|Provinsi)\s+([^,]+?)(?:\s+di\s+|$)", subject, flags=re.IGNORECASE)
    if match:
        return normalize_space(match.group(0).rstrip(" di"))
    return subject


def first_definition_term(source_doc: str) -> str | None:
    match = re.search(r"\d+\.\s*([^.;:]+?)\s+adalah\s+", source_doc, flags=re.IGNORECASE)
    if match:
        return normalize_space(match.group(1))
    return None


def first_sentence(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", normalize_space(text))
    return sentences[0].strip() if sentences else normalize_space(text)


def first_definition_answer(source_doc: str) -> str:
    match = re.search(r"\d+\.\s*([^.;:]+?\s+adalah\s+[^.]+\.)", source_doc, flags=re.IGNORECASE)
    if match:
        return normalize_space(match.group(1))
    return first_sentence(source_doc)


def build_question(row: dict) -> tuple[str, str, str] | None:
    source_doc = normalize_space(row.get("source_doc", ""))
    lowered = source_doc.lower()
    qtype = row.get("question_type_candidate", "article_qa")
    subject = title_subject(row)
    region = extract_region(row)

    if qtype == "definition":
        term = first_definition_term(source_doc) or subject
        return f"Apa yang dimaksud dengan {term} dalam aturan ini?", first_definition_answer(source_doc), "definition_realcase"

    if qtype == "sanction" or "dipidana" in lowered or "denda" in lowered:
        if "ancaman" in lowered or "menakut" in lowered:
            return "Apa sanksi hukum jika seseorang mengirim ancaman kekerasan atau pesan yang menakut-nakuti korban melalui media elektronik?", source_doc, "sanction_realcase"
        return f"Apa sanksi pidana atau denda yang diatur dalam {subject}?", source_doc, "sanction_realcase"

    if "informasi elektronik" in lowered or "sistem elektronik" in lowered:
        if "nama baik" in lowered or "kehormatan" in lowered:
            return "Apa aturan hukum jika seseorang menyerang kehormatan atau nama baik orang lain melalui sistem elektronik?", source_doc, "digital_law_realcase"
        if "ancaman kekerasan" in lowered:
            return "Apa aturan tentang pengiriman informasi elektronik yang berisi ancaman kekerasan kepada korban?", source_doc, "digital_law_realcase"
        return "Apa dasar hukum yang mengatur perbuatan melalui informasi elektronik atau sistem elektronik?", source_doc, "digital_law_realcase"

    if "kecamatan" in lowered and ("terdiri atas" in lowered or "terdiri dari" in lowered):
        return f"Berapa jumlah kecamatan di {region} dan apa saja kecamatannya?", source_doc, "regional_scope_realcase"

    if "ibu kota" in lowered or "ibukota" in lowered:
        return f"Di mana ibu kota {region} berkedudukan?", source_doc, "regional_scope_realcase"

    if "tanggal pembentukan" in lowered:
        return f"Kapan tanggal pembentukan {region} menurut aturan ini?", first_sentence(source_doc), "regional_scope_realcase"

    if "karakteristik" in lowered:
        return f"Apa karakteristik wilayah dan potensi utama {region}?", source_doc, "regional_scope_realcase"

    if "penyelenggaraan pemerintahan daerah" in lowered:
        return "Apa dasar pengaturan susunan dan tata cara penyelenggaraan pemerintahan daerah?", source_doc, "regional_governance_realcase"

    if "anggaran pendapatan dan belanja negara" in lowered or "apbn" in lowered:
        if "kementerian" in lowered or "lembaga" in lowered:
            return "Apa aturan mengenai anggaran kementerian dan lembaga dalam APBN?", source_doc, "state_budget_realcase"
        if "pemerintah daerah" in lowered:
            return "Apa aturan APBN yang berkaitan dengan pemerintah daerah?", source_doc, "state_budget_realcase"
        return "Apa ketentuan utama yang diatur dalam APBN?", source_doc, "state_budget_realcase"

    if "peraturan perundang-undangan" in lowered:
        return f"Apakah peraturan pelaksanaan lama masih berlaku setelah aturan tentang {region} berlaku?", source_doc, "legal_reference_realcase"

    return None


def build_rows(docs: list[dict], limit: int) -> list[dict]:
    rows = []
    seen_questions = set()
    for doc in docs:
        built = build_question(doc)
        if built is None:
            continue
        question, answer, question_type = built
        key = (question, doc.get("source_reference", ""))
        if key in seen_questions:
            continue
        seen_questions.add(key)
        rows.append(base_row(doc, question, answer, question_type))
        if len(rows) >= limit:
            break
    return rows


def write_split(path: Path, rows: list[dict], include_source_doc: bool) -> None:
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


def write_outputs(rows: list[dict], output: Path, split_dir: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    write_split(split_dir / "test_seen.jsonl", rows, include_source_doc=False)
    write_split(split_dir / "test_seen_with_context.jsonl", rows, include_source_doc=True)
    write_split(split_dir / "test_unseen.jsonl", rows, include_source_doc=False)
    write_split(split_dir / "test_unseen_with_context.jsonl", rows, include_source_doc=True)
    manifest = {
        "rows": len(rows),
        "laws": len({row["law_id"] for row in rows}),
        "question_types": sorted({row["question_type"] for row in rows}),
        "source_generation": "realcase_heuristic_from_doc_unit",
    }
    (split_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a small real-case Pasal.id eval set from doc units.")
    parser.add_argument("--input", default=str(DEFAULT_DOC_UNITS), help="Input doc_units JSONL")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output QA bank JSONL")
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR), help="Output split directory")
    parser.add_argument("--limit", type=int, default=60, help="Maximum rows to write")
    parser.add_argument("--max-source-chars", type=int, default=3000, help="Maximum source_doc character length")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    docs = [row for row in load_rows(Path(args.input)) if is_usable(row, args.max_source_chars)]
    rows = build_rows(docs, args.limit)
    if not rows:
        raise ValueError("No real-case rows generated. Check input filters.")
    write_outputs(rows, Path(args.output), Path(args.split_dir))


if __name__ == "__main__":
    main()
