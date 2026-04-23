import argparse
import json
from pathlib import Path


SOURCE_KEYS = ["source_type", "source_number", "source_year", "source_article"]


def parse_source_json(text: str) -> dict[str, str]:
    try:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            return {key: "" for key in SOURCE_KEYS}
        return {key: str(payload.get(key, "")).strip() for key in SOURCE_KEYS}
    except Exception:  # noqa: BLE001
        return {key: "" for key in SOURCE_KEYS}


def exact_source_match(prediction: dict[str, str], gold: dict[str, str]) -> int:
    return int(all(prediction[key] == gold[key] for key in SOURCE_KEYS))


def component_score(prediction: dict[str, str], gold: dict[str, str]) -> float:
    available = [key for key in SOURCE_KEYS if gold[key] != ""]
    if not available:
        return 0.0
    matches = sum(1 for key in available if prediction[key] == gold[key])
    return matches / len(available)


def evaluate_file(path: Path) -> dict[str, float | int]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {path}")

    exact_scores = []
    component_scores = []
    valid_json_scores = []
    per_key = {key: [] for key in SOURCE_KEYS}

    for row in rows:
        pred = parse_source_json(row["prediction"])
        gold = parse_source_json(row["gold"])
        valid_json_scores.append(int(any(pred.values())))
        exact_scores.append(exact_source_match(pred, gold))
        component_scores.append(component_score(pred, gold))
        for key in SOURCE_KEYS:
            if gold[key] != "":
                per_key[key].append(int(pred[key] == gold[key]))

    result = {
        "examples": len(rows),
        "valid_json_rate": sum(valid_json_scores) / len(valid_json_scores),
        "source_exact_match": sum(exact_scores) / len(exact_scores),
        "source_component_score": sum(component_scores) / len(component_scores),
    }
    for key, values in per_key.items():
        result[f"{key}_accuracy"] = sum(values) / len(values) if values else 0.0
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate source-prediction outputs.")
    parser.add_argument("--predictions", required=True, help="JSONL file containing gold and prediction fields")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate_file(Path(args.predictions))
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
