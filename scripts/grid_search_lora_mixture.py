#!/usr/bin/env python3

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

from lora_mlx.eval_legal import (
    citation_component_score,
    citation_exact_match,
    exact_match,
    split_answer_and_source,
    token_f1,
)
from lora_mlx.paths import DEFAULT_ADAPTERS_DIR, DEFAULT_DATA_DIR, DEFAULT_OUTPUTS_DIR


DEFAULT_BASIS = [
    DEFAULT_ADAPTERS_DIR / "adapters_pasalid_tinyllama_experiment.npz",
    DEFAULT_ADAPTERS_DIR / "adapters_pasalid_tinyllama_final.npz",
    DEFAULT_ADAPTERS_DIR / "adapters_pasalid_tinyllama_final_400.npz",
]
DEFAULT_DATA = DEFAULT_DATA_DIR / "pasalid" / "json_large_split" / "test_seen.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUTS_DIR / "reports" / "pasalid_lora_mixture_grid"


def parse_coefficients(value: str) -> tuple[float, ...]:
    coefficients = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    total = sum(coefficients)
    if total == 0:
        raise ValueError(f"Coefficient sum cannot be zero: {value}")
    return tuple(item / total for item in coefficients)


def default_candidates(count: int) -> list[tuple[float, ...]]:
    if count != 3:
        raise ValueError("Default candidates currently support exactly 3 basis adapters. Pass --candidates for another count.")
    return [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
        (1 / 3, 1 / 3, 1 / 3),
    ]


def validate_basis(basis_weights: list[dict]) -> list[str]:
    keys = list(basis_weights[0].keys())
    key_set = set(keys)
    for index, weights in enumerate(basis_weights[1:], start=1):
        if set(weights.keys()) != key_set:
            raise ValueError(f"Basis adapter {index} has different tensor keys")
        for key in keys:
            if tuple(weights[key].shape) != tuple(basis_weights[0][key].shape):
                raise ValueError(f"Basis adapter {index} tensor {key} has incompatible shape")
    return keys


def mix_adapter(basis_weights: list[dict], keys: list[str], coefficients: tuple[float, ...], output_path: Path) -> None:
    mixed = {}
    for key in keys:
        tensor = sum(float(weight) * basis[key] for weight, basis in zip(coefficients, basis_weights))
        mixed[key] = tensor.astype(mx.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(str(output_path), **mixed)


def run_export(model: str, adapter_path: Path, data_path: Path, output_path: Path, lora_layers: int, max_new_tokens: int, limit: int | None) -> None:
    command = [
        "python3",
        "-m",
        "lora_mlx.export",
        "--model",
        model,
        "--adapter-file",
        str(adapter_path),
        "--data",
        str(data_path),
        "--output",
        str(output_path),
        "--lora-layers",
        str(lora_layers),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    subprocess.run(command, check=True)


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


def group_key(row: dict, mode: str) -> str:
    try:
        payload = json.loads(row["gold"])
    except Exception:  # noqa: BLE001
        return "unknown"
    if mode == "law":
        return "/".join([str(payload.get("source_type", "")), str(payload.get("source_number", "")), str(payload.get("source_year", ""))])
    if mode == "article":
        return "/".join(
            [
                str(payload.get("source_type", "")),
                str(payload.get("source_number", "")),
                str(payload.get("source_year", "")),
                str(payload.get("source_article", "")),
            ]
        )
    raise ValueError(f"Unsupported group mode: {mode}")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grid-search LoRA mixture coefficients and compute global/per-group oracle metrics.")
    parser.add_argument("--basis", nargs="+", default=[str(path) for path in DEFAULT_BASIS], help="Basis adapter .npz files")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Evaluation JSONL data")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for adapters, predictions, and reports")
    parser.add_argument("--model", default="mlx_model", help="Base model path")
    parser.add_argument("--lora-layers", type=int, default=4, help="LoRA layers")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max generation tokens")
    parser.add_argument("--limit", type=int, default=None, help="Optional evaluation limit")
    parser.add_argument("--group-by", choices=["law", "article"], default="law", help="Oracle routing group key")
    parser.add_argument("--candidates", nargs="*", default=None, help="Coefficient candidates like '1,0,0' '0.5,0.5,0'")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    adapter_dir = output_dir / "adapters"
    prediction_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    basis_paths = [Path(path) for path in args.basis]
    basis_weights = [mx.load(str(path)) for path in basis_paths]
    keys = validate_basis(basis_weights)
    candidates = [parse_coefficients(value) for value in args.candidates] if args.candidates else default_candidates(len(basis_paths))

    candidate_rows = []
    for index, coefficients in enumerate(candidates):
        adapter_path = adapter_dir / f"candidate_{index}.npz"
        prediction_path = prediction_dir / f"candidate_{index}.jsonl"
        mix_adapter(basis_weights, keys, coefficients, adapter_path)
        run_export(
            model=args.model,
            adapter_path=adapter_path,
            data_path=Path(args.data),
            output_path=prediction_path,
            lora_layers=args.lora_layers,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
        )
        rows = load_jsonl(prediction_path)
        candidate_rows.append(rows)

    global_metrics = []
    for index, rows in enumerate(candidate_rows):
        metrics = summarize(rows)
        metrics["candidate_index"] = index
        metrics["coefficients"] = list(candidates[index])
        global_metrics.append(metrics)

    groups = defaultdict(list)
    for row_index, row in enumerate(candidate_rows[0]):
        groups[group_key(row, args.group_by)].append(row_index)

    per_group = []
    oracle_rows = []
    for group, indices in groups.items():
        best_index = None
        best_score = -1.0
        best_metrics = None
        for candidate_index, rows in enumerate(candidate_rows):
            group_rows = [rows[index] for index in indices]
            metrics = summarize(group_rows)
            score = metrics["answer_f1"]
            if score > best_score:
                best_score = score
                best_index = candidate_index
                best_metrics = metrics
        oracle_rows.extend(candidate_rows[best_index][index] for index in indices)
        per_group.append(
            {
                "group": group,
                "examples": len(indices),
                "best_candidate_index": best_index,
                "best_coefficients": list(candidates[best_index]),
                "best_metrics": best_metrics,
            }
        )

    oracle_predictions_path = output_dir / "oracle_predictions.jsonl"
    with oracle_predictions_path.open("w") as handle:
        for row in sorted(oracle_rows, key=lambda item: item["index"]):
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    report = {
        "basis": [str(path) for path in basis_paths],
        "data": str(args.data),
        "group_by": args.group_by,
        "candidate_count": len(candidates),
        "global_metrics": global_metrics,
        "best_global": max(global_metrics, key=lambda item: item["answer_f1"]),
        "oracle_metrics": summarize(oracle_rows),
        "per_group": sorted(per_group, key=lambda item: item["group"]),
        "oracle_predictions": str(oracle_predictions_path),
    }
    report_path = output_dir / "grid_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
