#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


REFERENCE_RE = re.compile(
    r"Referensi:\s*(UU)\s+No\.\s*([A-Za-z0-9]+)\s+Tahun\s+(\d{4}),\s+Pasal\s+([^\n]+)",
    flags=re.IGNORECASE,
)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_prompt_reference(prompt: str) -> dict[str, str]:
    match = REFERENCE_RE.search(prompt)
    if not match:
        return {"source_type": "", "source_number": "", "source_year": "", "source_article": ""}
    return {
        "source_type": match.group(1).upper(),
        "source_number": match.group(2).strip(),
        "source_year": match.group(3).strip(),
        "source_article": normalize_space(match.group(4)),
    }


def extract_json_payload(text: str) -> dict | None:
    stripped = text.strip()
    candidates = [stripped]
    if "{" in stripped and "}" in stripped:
        candidates.append(stripped[stripped.find("{") : stripped.rfind("}") + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def clean_answer_text(text: str) -> str:
    answer = text.strip()
    if answer.startswith("Jawab:"):
        answer = answer.removeprefix("Jawab:").strip()
    for marker in ["\nReferensi:", "\nInstruksi:", "\nQ:", " Referensi:", " Instruksi:"]:
        marker_index = answer.find(marker)
        if marker_index != -1:
            answer = answer[:marker_index].strip()
    return normalize_space(answer)


def build_constrained_prediction(row: dict) -> str:
    prompt_ref = parse_prompt_reference(row.get("prompt", ""))
    prediction = row.get("prediction", row.get("pred", ""))
    payload = extract_json_payload(prediction)

    if payload and "answer" in payload:
        answer = clean_answer_text(str(payload.get("answer", "")))
    else:
        answer = clean_answer_text(prediction)

    constrained = {
        "answer": answer,
        "source_type": prompt_ref["source_type"],
        "source_number": prompt_ref["source_number"],
        "source_year": prompt_ref["source_year"],
        "source_article": prompt_ref["source_article"],
    }
    return json.dumps(constrained, ensure_ascii=False, separators=(",", ":"))


def postprocess(input_path: Path, output_path: Path) -> int:
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for row in rows:
            row["raw_prediction"] = row.get("prediction", row.get("pred", ""))
            row["prediction"] = build_constrained_prediction(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-process Pasal.id natural QA predictions into constrained answer+source JSON.")
    parser.add_argument("--input", required=True, help="Input prediction JSONL")
    parser.add_argument("--output", required=True, help="Output constrained prediction JSONL")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    count = postprocess(Path(args.input), Path(args.output))
    print(json.dumps({"rows": count, "output": args.output}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
