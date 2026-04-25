#!/usr/bin/env python3

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


STRICT_FAILURE_LABELS = {"factually-wrong", "unsupported-answer", "source-wrong", "source-missing"}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def question_from_prompt(prompt: str) -> str:
    match = re.search(r"\nQ:\s*(.*?)\nA:\s*$", prompt, flags=re.DOTALL)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def categorize(row: dict, model_key: str) -> list[str]:
    labels = set(row.get(f"{model_key}_error_labels", []))
    question = question_from_prompt(row.get("prompt", "")).lower()
    reason = str(row.get("reason", "")).lower()
    prediction = str(row.get(f"{model_key}_prediction", ""))
    prediction_lower = prediction.lower()
    text = " ".join([question, reason, prediction_lower])
    categories = []

    valid_json_like = prediction.strip().startswith("{") and prediction.strip().endswith("}")
    if "source-wrong" in labels or "source-missing" in labels or not valid_json_like:
        categories.append("source/format error")
    if "too-extractive" in labels or "menyalin" in text or "copy" in text or "ekstraktif" in text:
        categories.append("too extractive")
    if "unnatural" in labels or "instruksi" in prediction_lower or "referensi:" in prediction_lower or "q:" in prediction_lower:
        categories.append("prompt/instruction echo or unnatural")
    if any(marker in text for marker in ["terlalu pendek", "hanya", "tidak menjawab", "bukan", "kurang lengkap", "partial", "sebagian"]):
        categories.append("incomplete or wrong focus")
    if any(marker in text for marker in ["berapa", "jumlah", "kecamatan", "daftar", "lembaga", "provinsi", "wilayah", "lembaran negara"]):
        categories.append("entity/count/list confusion")
    if any(marker in text for marker in ["masih berlaku", "tidak berlaku", "dicabut", "batal", "gugur", "negasi"]):
        categories.append("polarity/transition confusion")
    if "factually-wrong" in labels or "unsupported-answer" in labels:
        categories.append("substantive factual/evidence error")

    return categories or ["other"]


def summarize(path: Path, model_key: str) -> dict:
    rows = load_rows(path)
    failure_rows = []
    strict_failure_rows = []
    category_counts = Counter()
    label_counts = Counter()
    examples = defaultdict(list)

    for row in rows:
        labels = set(row.get(f"{model_key}_error_labels", []))
        label_counts.update(labels)
        is_strict_failure = bool(labels & STRICT_FAILURE_LABELS)
        is_failure = is_strict_failure or row.get(model_key, {}).get("factual_correctness", 2) < 2 or row.get(model_key, {}).get("evidence_support", 2) < 2
        if not is_failure:
            continue
        failure_rows.append(row)
        if is_strict_failure:
            strict_failure_rows.append(row)
        categories = categorize(row, model_key)
        category_counts.update(categories)
        for category in categories:
            if len(examples[category]) < 5:
                examples[category].append(
                    {
                        "index": row.get("index"),
                        "question": question_from_prompt(row.get("prompt", "")),
                        "labels": sorted(labels),
                        "reason": row.get("reason", ""),
                    }
                )

    return {
        "input": str(path),
        "model_key": model_key,
        "rows": len(rows),
        "failure_rows": len(failure_rows),
        "failure_rate": len(failure_rows) / len(rows) if rows else 0.0,
        "strict_failure_rows": len(strict_failure_rows),
        "strict_failure_rate": len(strict_failure_rows) / len(rows) if rows else 0.0,
        "label_counts": dict(label_counts),
        "category_counts": dict(category_counts),
        "examples": dict(examples),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Pasal.id pairwise review failure categories.")
    parser.add_argument("--input", required=True, help="Pairwise review JSONL")
    parser.add_argument("--output", required=True, help="Output summary JSON")
    parser.add_argument("--model-key", choices=["B", "D"], default="D", help="Which reviewed model to audit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize(Path(args.input), args.model_key)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
