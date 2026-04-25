#!/usr/bin/env python3

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from lora_mlx.eval_legal import split_answer_and_source


INDONESIAN_NUMBERS = {
    "1": "satu",
    "2": "dua",
    "3": "tiga",
    "4": "empat",
    "5": "lima",
    "6": "enam",
    "7": "tujuh",
    "8": "delapan",
    "9": "sembilan",
    "10": "sepuluh",
    "11": "sebelas",
    "12": "dua belas",
    "13": "tiga belas",
    "14": "empat belas",
    "15": "lima belas",
    "16": "enam belas",
    "17": "tujuh belas",
    "18": "delapan belas",
    "19": "sembilan belas",
    "20": "dua puluh",
    "21": "dua puluh satu",
    "22": "dua puluh dua",
    "23": "dua puluh tiga",
    "24": "dua puluh empat",
    "25": "dua puluh lima",
}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("propinsi", "provinsi")
    text = re.sub(r"\bi\s*1\b", "11", text)
    text = re.sub(r"\bl\s*4\b", "14", text)
    text = re.sub(r"\bl(?=\d)", "1", text)
    text = re.sub(r"(?<=\d)l\b", "1", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return normalize_space(text)


def contains_phrase(text: str, phrase: str) -> bool:
    return normalize_text(phrase) in normalize_text(text)


def contains_number(text: str, number: str, word: str = "") -> bool:
    normalized = normalize_text(text)
    if re.search(rf"(^|\s){re.escape(number)}(\s|$)", normalized):
        return True
    if word and normalize_text(word) in normalized:
        return True
    default_word = INDONESIAN_NUMBERS.get(number)
    return bool(default_word and default_word in normalized)


def parse_prompt(prompt: str) -> dict[str, str]:
    source_doc = ""
    reference = ""
    question = ""
    if prompt.startswith("Dokumen sumber: "):
        source_doc = prompt.split("Dokumen sumber: ", 1)[1].split("\nReferensi: ", 1)[0]
    reference_match = re.search(r"\nReferensi:\s*(.*?)\n", prompt)
    if reference_match:
        reference = normalize_space(reference_match.group(1))
    question_match = re.search(r"\nQ:\s*(.*?)\nA:\s*$", prompt, flags=re.DOTALL)
    if question_match:
        question = normalize_space(question_match.group(1))
    return {"source_doc": source_doc, "reference": reference, "question": question}


def normalize_count(raw: str) -> str:
    return normalize_text(raw).replace(" ", "")


def extract_count(source_doc: str) -> tuple[str, str] | None:
    match = re.search(r"terdiri atas\s+([0-9Il\s]+)\s*\(([^)]+)\)\s+Kecamatan", source_doc, flags=re.IGNORECASE)
    if not match:
        return None
    return normalize_count(match.group(1)), normalize_space(match.group(2))


def extract_old_province(source_doc: str) -> str:
    match = re.search(r"lingkungan\s+Daerah\s+Propinsi\s+([^(.]+)", source_doc, flags=re.IGNORECASE)
    return normalize_space(match.group(1)) if match else ""


def extract_current_province(source_doc: str) -> str:
    match = re.search(r"berada\s+di\s+wilayah\s+Provinsi\s+([^.;]+?)(?:\s+yang|\.|$)", source_doc, flags=re.IGNORECASE)
    return normalize_space(match.group(1)) if match else ""


def extract_gazette(source_doc: str) -> tuple[str, str] | None:
    match = re.search(r"Lembaran(?:-Negara| Negara)?(?:\s+Tahun\s+(\d{4}))?\s+Nomor\s+([0-9Il]+)(?:\s+Tahun\s+(\d{4}))?", source_doc, flags=re.IGNORECASE)
    if not match:
        return None
    return normalize_count(match.group(2)), match.group(1) or match.group(3) or ""


def split_list_items(answer: str) -> list[str]:
    answer = re.sub(r"^.*?(?:yaitu|adalah|mencakup|meliputi)\s+", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\b(?:Kecamatan|wilayah|kabupaten|kota)\b", " ", answer, flags=re.IGNORECASE)
    parts = re.split(r",|\bdan\b", answer)
    items = []
    for part in parts:
        normalized = normalize_text(part)
        normalized = re.sub(r"\b(?:berdasarkan|pasal|uu|no|nomor|tahun|tersebut|itu|dalam|ketentuan|ini)\b.*$", "", normalized).strip()
        if len(normalized.split()) >= 1 and not normalized.isdigit():
            items.append(normalized)
    return [item for item in items if item]


def list_recall(prediction: str, gold_answer: str) -> tuple[int, int, float]:
    expected = split_list_items(gold_answer)
    if not expected:
        return 0, 0, 0.0
    normalized_prediction = normalize_text(prediction)
    matched = sum(1 for item in expected if item in normalized_prediction)
    return matched, len(expected), matched / len(expected)


def build_checks(row: dict) -> list[dict]:
    prompt = parse_prompt(row.get("prompt", ""))
    source_doc = prompt["source_doc"]
    question = prompt["question"].lower()
    gold_answer, _ = split_answer_and_source(row.get("gold", ""))
    checks = []

    count = extract_count(source_doc)
    if count and any(marker in question for marker in ["berapa", "jumlah", "ada berapa", "terbagi"]):
        checks.append({"slot": "kecamatan_count", "expected": count})

    if count and any(marker in question for marker in ["apa saja", "sebutkan", "nama kecamatan", "kecamatan apa saja"]):
        checks.append({"slot": "kecamatan_list", "gold_answer": gold_answer})

    old_province = extract_old_province(source_doc)
    if old_province and "provinsi" in question and any(marker in question for marker in ["pertama kali", "dibentuk", "awal"]):
        checks.append({"slot": "old_formation_province", "expected": old_province})

    current_province = extract_current_province(source_doc)
    if current_province and "provinsi" in question and not any(marker in question for marker in ["pertama kali", "awal"]):
        checks.append({"slot": "current_province", "expected": current_province})

    gazette = extract_gazette(source_doc)
    if gazette and "lembaran negara" in question:
        checks.append({"slot": "gazette_reference", "expected": gazette})

    lowered_source = source_doc.lower()
    asks_transition_status = any(
        marker in question
        for marker in [
            "masih berlaku",
            "tidak berlaku",
            "dicabut",
            "nasib aturan",
            "aturan lama",
            "aturan turunan",
            "otomatis gugur",
            "otomatis tidak dipakai",
            "status ketentuan",
        ]
    )
    if asks_transition_status and "dicabut" in lowered_source and "dinyatakan tidak berlaku" in lowered_source:
        checks.append({"slot": "repeal_status", "expected": "dicabut dan tidak berlaku"})
    if asks_transition_status and "masih tetap berlaku" in lowered_source and "sepanjang tidak bertentangan" in lowered_source:
        checks.append({"slot": "transition_validity", "expected": "masih berlaku sepanjang tidak bertentangan"})

    return checks


def evaluate_check(check: dict, prediction_answer: str) -> tuple[bool, dict]:
    slot = check["slot"]
    if slot == "kecamatan_count":
        number, word = check["expected"]
        return contains_number(prediction_answer, number, word), {"number": number, "word": word}
    if slot == "kecamatan_list":
        matched, total, recall = list_recall(prediction_answer, check["gold_answer"])
        return recall >= 0.8, {"matched_items": matched, "total_items": total, "item_recall": recall}
    if slot in {"old_formation_province", "current_province"}:
        expected = check["expected"]
        return contains_phrase(prediction_answer, expected), {"expected": expected}
    if slot == "gazette_reference":
        number, year = check["expected"]
        ok = "lembaran" in normalize_text(prediction_answer) and contains_number(prediction_answer, number)
        if year:
            ok = ok and contains_number(prediction_answer, year)
        return ok, {"number": number, "year": year}
    if slot == "repeal_status":
        normalized = normalize_text(prediction_answer)
        ok = "dicabut" in normalized and "tidak berlaku" in normalized
        return ok, {"expected": check["expected"]}
    if slot == "transition_validity":
        normalized = normalize_text(prediction_answer)
        ok = "berlaku" in normalized and "tidak bertentangan" in normalized
        return ok, {"expected": check["expected"]}
    return False, {}


def evaluate(path: Path) -> dict:
    rows = load_rows(path)
    by_slot = defaultdict(lambda: {"total": 0, "correct": 0})
    examples = defaultdict(list)
    total = 0
    correct = 0

    for row in rows:
        prediction_answer, _ = split_answer_and_source(row.get("prediction", row.get("pred", "")))
        prompt = parse_prompt(row.get("prompt", ""))
        for check in build_checks(row):
            ok, detail = evaluate_check(check, prediction_answer)
            slot = check["slot"]
            total += 1
            correct += int(ok)
            by_slot[slot]["total"] += 1
            by_slot[slot]["correct"] += int(ok)
            if not ok and len(examples[slot]) < 5:
                examples[slot].append(
                    {
                        "index": row.get("index"),
                        "question": prompt["question"],
                        "expected": detail,
                        "prediction_answer": prediction_answer,
                    }
                )

    slots = {}
    for slot, values in sorted(by_slot.items()):
        slots[slot] = {
            "total": values["total"],
            "correct": values["correct"],
            "accuracy": values["correct"] / values["total"] if values["total"] else 0.0,
        }
    return {
        "input": str(path),
        "rows": len(rows),
        "slot_checks": total,
        "slot_correct": correct,
        "slot_accuracy": correct / total if total else 0.0,
        "slots": slots,
        "failure_examples": dict(examples),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate targeted factual slots in Pasal.id natural legal QA predictions.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL with prompt/gold/prediction fields")
    parser.add_argument("--output", default="", help="Optional output JSON path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate(Path(args.predictions))
    text = json.dumps(summary, ensure_ascii=True, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
