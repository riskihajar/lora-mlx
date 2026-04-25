#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path

from lora_mlx.paths import DEFAULT_DATA_DIR, DEFAULT_PREDICTIONS_DIR


DEFAULT_EXPERIMENT_DIR = DEFAULT_DATA_DIR / "pasalid" / "experiment_split"
DEFAULT_EXPORT_DIR = DEFAULT_PREDICTIONS_DIR / "pasalid_experiment"


MODEL_PRESETS = {
    "tinyllama": {
        "model": "mlx_model",
        "adapter": "outputs/adapters/adapters_pasalid_tinyllama_experiment.npz",
    },
    "tinyllama_final": {
        "model": "mlx_model",
        "adapter": "outputs/adapters/adapters_pasalid_tinyllama_final.npz",
    },
    "tinyllama_final_400": {
        "model": "mlx_model",
        "adapter": "outputs/adapters/adapters_pasalid_tinyllama_final_400.npz",
    },
    "tinyllama_native_expanded": {
        "model": "mlx_model",
        "adapter": "outputs/adapters/adapters_pasalid_tinyllama_native_expanded.npz",
    },
    "tinyllama_native_expanded_clean": {
        "model": "mlx_model",
        "adapter": "outputs/adapters/adapters_pasalid_tinyllama_native_expanded_clean.npz",
    },
    "tinyllama_natural_legal": {
        "model": "mlx_model",
        "adapter": "outputs/adapters/adapters_pasalid_tinyllama_natural_legal.npz",
    },
    "tinyllama_hyperproto_zero": {
        "model": "mlx_model",
        "adapter": "outputs/adapters/adapters_pasalid_hyperproto_tinyllama_zero.npz",
    },
    "tinyllama_hyperproto_hash": {
        "model": "mlx_model",
        "adapter": "outputs/adapters/adapters_pasalid_hyperproto_tinyllama_hash.npz",
    },
    "tinyllama_hyperproto_mixture": {
        "model": "mlx_model",
        "adapter": "outputs/adapters/adapters_pasalid_hyperproto_tinyllama_mixture.npz",
    },
    "mistral_q4": {
        "model": "mlx_model_mistral_q4",
        "adapter": "outputs/adapters/adapters_pasalid_mistral_q4_experiment.npz",
    },
    "mistral_q4_long": {
        "model": "mlx_model_mistral_q4",
        "adapter": "outputs/adapters/adapters_pasalid_mistral_q4_experiment_long.npz",
    },
    "qwen3": {
        "model": "mlx-community/Qwen3-4B-8bit",
        "adapter": "outputs/adapters/adapters_pasalid_qwen3_experiment.npz",
    },
}


def run_export(model: str, data_path: Path, output_path: Path, adapter_file: str | None, lora_layers: int, max_new_tokens: int, limit: int | None) -> None:
    command = [
        "python3",
        "-m",
        "lora_mlx.export",
        "--model",
        model,
        "--data",
        str(data_path),
        "--output",
        str(output_path),
        "--lora-layers",
        str(lora_layers),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    if adapter_file:
        command.extend(["--adapter-file", adapter_file])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export A/B/C Pasal.id experiment predictions, with optional D.")
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), default="tinyllama", help="Model preset")
    parser.add_argument("--split", choices=["seen", "unseen"], default="seen", help="Which experiment split to export")
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR), help="Experiment split directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_EXPORT_DIR), help="Prediction export directory")
    parser.add_argument("--lora-layers", type=int, default=4, help="Number of LoRA layers")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max generation tokens")
    parser.add_argument("--limit", type=int, default=None, help="Optional export limit")
    parser.add_argument("--include-d", action="store_true", help="Also export condition D: adapter with source context")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    preset = MODEL_PRESETS[args.preset]
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "seen":
        no_context = experiment_dir / "test_seen.jsonl"
        with_context = experiment_dir / "test_seen_with_context.jsonl"
    else:
        no_context = experiment_dir / "test_unseen.jsonl"
        with_context = experiment_dir / "test_unseen_with_context.jsonl"

    exports = {
        "A_base_no_context": {
            "data": no_context,
            "adapter": None,
        },
        "B_base_with_context": {
            "data": with_context,
            "adapter": None,
        },
        "C_adapter_no_context": {
            "data": no_context,
            "adapter": preset["adapter"],
        },
    }
    if args.include_d:
        exports["D_adapter_with_context"] = {
            "data": with_context,
            "adapter": preset["adapter"],
        }

    results = {}
    for label, config in exports.items():
        output_path = output_dir / f"{args.preset}_{args.split}_{label}.jsonl"
        run_export(
            model=preset["model"],
            data_path=config["data"],
            output_path=output_path,
            adapter_file=config["adapter"],
            lora_layers=args.lora_layers,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
        )
        results[label] = str(output_path)

    print(json.dumps({"preset": args.preset, "split": args.split, "exports": results}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
