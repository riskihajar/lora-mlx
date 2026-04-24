#!/usr/bin/env python3

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from lora_mlx.eval_legal import (
    citation_component_score,
    citation_exact_match,
    exact_match,
    split_answer_and_source,
    token_f1,
)
from lora_mlx.paths import DEFAULT_DATA_DIR, DEFAULT_OUTPUTS_DIR


DEFAULT_DOC_UNITS = DEFAULT_DATA_DIR / "pasalid" / "doc_units.jsonl"
DEFAULT_TRAIN_REPORT = DEFAULT_OUTPUTS_DIR / "reports" / "pasalid_lora_mixture_grid_json_large_seen" / "grid_report.json"
DEFAULT_EVAL_REPORT = DEFAULT_TRAIN_REPORT
DEFAULT_OUTPUT = DEFAULT_OUTPUTS_DIR / "reports" / "pasalid_lora_mixture_router" / "router_report.json"


def parse_group(group: str) -> tuple[str, str, str]:
    parts = group.split("/")
    if len(parts) != 3:
        raise ValueError(f"Expected group format TYPE/NUMBER/YEAR, got {group}")
    return parts[0], parts[1], parts[2]


def law_group_from_id(law_id: str) -> str | None:
    match = re.search(r"/act/([^/]+)/(\d{4})/([^/]+)$", law_id)
    if not match:
        return None
    law_type, year, number = match.groups()
    return f"{law_type.upper()}/{number}/{year}"


def load_doc_features(path: Path) -> dict[str, dict[str, float]]:
    grouped = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        group = law_group_from_id(str(row.get("law_id", "")))
        if group:
            grouped[group].append(row)

    features = {}
    for group, rows in grouped.items():
        _, number, year = parse_group(group)
        docs = [str(row.get("source_doc", "")) for row in rows]
        text = "\n".join(docs).lower()
        qtypes = Counter(str(row.get("question_type_candidate", "")) for row in rows)
        features[group] = {
            "law_number": float(number) if number.isdigit() else 0.0,
            "law_year": float(year) if year.isdigit() else 0.0,
            "article_count": float(len(rows)),
            "avg_doc_chars": float(sum(len(doc) for doc in docs) / max(len(docs), 1)),
            "has_sanction": float("pidana" in text or "denda" in text),
            "has_transition": float("tetap berlaku" in text or "dicabut" in text),
            "article_qa_ratio": qtypes.get("article_qa", 0) / max(len(rows), 1),
        }
    return features


FEATURE_NAMES = [
    "law_number",
    "law_year",
    "article_count",
    "avg_doc_chars",
    "has_sanction",
    "has_transition",
    "article_qa_ratio",
]


def feature_vector(group: str, doc_features: dict[str, dict[str, float]]) -> np.ndarray:
    values = doc_features.get(group)
    if values is None:
        _, number, year = parse_group(group)
        values = {
            "law_number": float(number) if number.isdigit() else 0.0,
            "law_year": float(year) if year.isdigit() else 0.0,
            "article_count": 0.0,
            "avg_doc_chars": 0.0,
            "has_sanction": 0.0,
            "has_transition": 0.0,
            "article_qa_ratio": 0.0,
        }
    return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float64)


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std, mean, std


def train_softmax(x: np.ndarray, y: np.ndarray, class_count: int, steps: int, lr: float, l2: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    weights = np.zeros((x.shape[1], class_count), dtype=np.float64)
    y_onehot = np.eye(class_count)[y]
    for _ in range(steps):
        logits = x @ weights
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        grad = x.T @ (probs - y_onehot) / x.shape[0]
        grad[:-1] += l2 * weights[:-1]
        weights -= lr * grad
    return weights[:-1], weights[-1]


def predict(x: np.ndarray, mean: np.ndarray, std: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    x_std = (x - mean) / std
    logits = x_std @ weights + bias
    return logits.argmax(axis=1)


def row_group(row: dict) -> str:
    _, gold_source = split_answer_and_source(row["gold"])
    try:
        payload = json.loads(row["gold"])
        return f"{payload.get('source_type', '')}/{payload.get('source_number', '')}/{payload.get('source_year', '')}"
    except Exception:  # noqa: BLE001
        return gold_source or "unknown"


def row_metrics(row: dict) -> dict[str, float]:
    pred_answer, pred_source = split_answer_and_source(row["prediction"])
    gold_answer, gold_source = split_answer_and_source(row["gold"])
    return {
        "answer_em": exact_match(pred_answer, gold_answer),
        "answer_f1": token_f1(pred_answer, gold_answer),
        "citation_em": citation_exact_match(pred_source, gold_source),
        "citation_component_score": citation_component_score(pred_source, gold_source),
    }


def summarize(rows: list[dict]) -> dict[str, float]:
    metrics = [row_metrics(row) for row in rows]
    return {
        "examples": len(rows),
        "answer_em": sum(item["answer_em"] for item in metrics) / len(metrics),
        "answer_f1": sum(item["answer_f1"] for item in metrics) / len(metrics),
        "citation_em": sum(item["citation_em"] for item in metrics) / len(metrics),
        "citation_component_score": sum(item["citation_component_score"] for item in metrics) / len(metrics),
    }


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_candidate_rows(report_path: Path, candidate_count: int) -> list[list[dict]]:
    prediction_dir = report_path.parent / "predictions"
    return [load_jsonl(prediction_dir / f"candidate_{index}.jsonl") for index in range(candidate_count)]


def route_rows(eval_report: dict, eval_report_path: Path, doc_features: dict[str, dict[str, float]], mean: np.ndarray, std: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> tuple[list[dict], dict[str, int]]:
    candidate_count = int(eval_report["candidate_count"])
    candidate_rows = load_candidate_rows(eval_report_path, candidate_count)
    groups = sorted({row_group(row) for row in candidate_rows[0]})
    x = np.stack([feature_vector(group, doc_features) for group in groups])
    pred_classes = predict(x, mean, std, weights, bias)
    group_to_candidate = {group: int(candidate) for group, candidate in zip(groups, pred_classes)}
    routed = []
    for row_index, row in enumerate(candidate_rows[0]):
        group = row_group(row)
        routed.append(candidate_rows[group_to_candidate[group]][row_index])
    return routed, group_to_candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tiny law-feature router for LoRA mixture candidates.")
    parser.add_argument("--train-report", default=str(DEFAULT_TRAIN_REPORT), help="Grid report used as router supervision")
    parser.add_argument("--eval-report", default=str(DEFAULT_EVAL_REPORT), help="Grid report to route/evaluate")
    parser.add_argument("--doc-units", default=str(DEFAULT_DOC_UNITS), help="Doc units JSONL for law-level features")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output router report JSON")
    parser.add_argument("--steps", type=int, default=1000, help="Softmax training steps")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Softmax learning rate")
    parser.add_argument("--l2", type=float, default=0.01, help="L2 regularization")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_report_path = Path(args.train_report)
    eval_report_path = Path(args.eval_report)
    train_report = json.loads(train_report_path.read_text())
    eval_report = json.loads(eval_report_path.read_text())
    doc_features = load_doc_features(Path(args.doc_units))

    train_groups = [row["group"] for row in train_report["per_group"]]
    labels = np.array([int(row["best_candidate_index"]) for row in train_report["per_group"]], dtype=np.int64)
    class_count = int(train_report["candidate_count"])
    x = np.stack([feature_vector(group, doc_features) for group in train_groups])
    x_std, mean, std = standardize(x)
    weights, bias = train_softmax(x_std, labels, class_count, args.steps, args.learning_rate, args.l2)

    train_pred = predict(x, mean, std, weights, bias)
    routed_rows, group_to_candidate = route_rows(eval_report, eval_report_path, doc_features, mean, std, weights, bias)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    routed_path = output_path.with_name(output_path.stem + "_predictions.jsonl")
    with routed_path.open("w") as handle:
        for row in routed_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    report = {
        "train_report": str(train_report_path),
        "eval_report": str(eval_report_path),
        "feature_names": FEATURE_NAMES,
        "train_groups": train_groups,
        "train_labels": labels.tolist(),
        "train_predictions": train_pred.tolist(),
        "train_accuracy": float((train_pred == labels).mean()),
        "group_to_candidate": group_to_candidate,
        "coefficients_by_candidate": [item["coefficients"] for item in train_report["global_metrics"]],
        "router_metrics": summarize(routed_rows),
        "best_global": eval_report["best_global"],
        "oracle_metrics": eval_report["oracle_metrics"],
        "routed_predictions": str(routed_path),
        "note": "Tiny router over law-level features; intended as feasibility check, not final hypernetwork.",
    }
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
