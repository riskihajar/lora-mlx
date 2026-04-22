import argparse
import json
import re
from collections import Counter
from pathlib import Path


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def normalize_tokens(text: str) -> list[str]:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([(),.:;])\s*", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().split()


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_tokens(prediction)
    gold_tokens = normalize_tokens(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def split_answer_and_source(text: str) -> tuple[str, str]:
    normalized = normalize_text(text)
    if normalized.startswith("{") and normalized.endswith("}"):
        try:
            payload = json.loads(normalized)
            if isinstance(payload, dict) and "answer" in payload:
                if "source" in payload:
                    return normalize_text(str(payload["answer"])), normalize_text(str(payload["source"]))
                source_parts = [
                    str(payload.get("source_type", "")).strip(),
                    str(payload.get("source_number", "")).strip(),
                    str(payload.get("source_year", "")).strip(),
                    str(payload.get("source_article", "")).strip(),
                ]
                source_text = " ".join(part for part in source_parts if part)
                return normalize_text(str(payload["answer"])), normalize_text(source_text)
        except Exception:  # noqa: BLE001
            pass

    match = re.search(r"(?:^|\s)Sumber:\s*(.+)$", text, flags=re.IGNORECASE)
    if not match:
        return normalize_text(text), ""
    answer = normalize_text(text[: match.start()].strip())
    source = normalize_text(match.group(1).strip())
    return answer, source


def exact_match(left: str, right: str) -> int:
    return int(normalize_text(left) == normalize_text(right))


def parse_reference(text: str) -> dict[str, str]:
    normalized = normalize_text(text)
    result = {"type": "", "number": "", "year": "", "article": ""}

    parts = normalized.split()
    if len(parts) == 4 and parts[0].upper() == "UU":
        result["type"] = "UU"
        result["number"] = parts[1]
        result["year"] = parts[2]
        result["article"] = parts[3]
        return result

    law_match = re.search(r"\b(UU|Undang-Undang)\b\s*(?:No\.?|Nomor)?\s*([A-Za-z0-9]+)\s*Tahun\s*(\d{4})", normalized, flags=re.IGNORECASE)
    if law_match:
        result["type"] = "UU"
        result["number"] = law_match.group(2)
        result["year"] = law_match.group(3)

    article_match = re.search(r"Pasal\s+([A-Za-z0-9]+(?:\s+huruf\s+[a-z])?)", normalized, flags=re.IGNORECASE)
    if article_match:
        result["article"] = article_match.group(1)

    return result


def citation_exact_match(prediction_source: str, gold_source: str) -> int:
    return int(parse_reference(prediction_source) == parse_reference(gold_source) and prediction_source != "" and gold_source != "")


def citation_component_score(prediction_source: str, gold_source: str) -> float:
    pred = parse_reference(prediction_source)
    gold = parse_reference(gold_source)
    components = ["type", "number", "year", "article"]
    available = [component for component in components if gold[component]]
    if not available:
        return 0.0
    matches = sum(1 for component in available if pred[component] == gold[component])
    return matches / len(available)


def evaluate_file(path: Path) -> dict[str, float | int]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {path}")

    answer_ems = []
    answer_f1s = []
    citation_ems = []
    citation_scores = []

    for row in rows:
        prediction = row["prediction"] if "prediction" in row else row.get("pred", "")
        gold = row["gold"]
        pred_answer, pred_source = split_answer_and_source(prediction)
        gold_answer, gold_source = split_answer_and_source(gold)

        answer_ems.append(exact_match(pred_answer, gold_answer))
        answer_f1s.append(token_f1(pred_answer, gold_answer))
        citation_ems.append(citation_exact_match(pred_source, gold_source))
        citation_scores.append(citation_component_score(pred_source, gold_source))

    return {
        "examples": len(rows),
        "answer_em": sum(answer_ems) / len(answer_ems),
        "answer_f1": sum(answer_f1s) / len(answer_f1s),
        "citation_em": sum(citation_ems) / len(citation_ems),
        "citation_component_score": sum(citation_scores) / len(citation_scores),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate legal QA predictions with separate answer and citation scoring.")
    parser.add_argument("--predictions", required=True, help="JSONL file containing gold and prediction fields")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate_file(Path(args.predictions))
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
