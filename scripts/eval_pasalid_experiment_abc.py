#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path

from lora_mlx.paths import DEFAULT_DATA_DIR


DEFAULT_EXPERIMENT_DIR = DEFAULT_DATA_DIR / "pasalid" / "experiment_split"


MODEL_PRESETS = {
    "tinyllama": {
        "model": "mlx_model",
        "adapter": "outputs/adapters/adapters_pasalid_tinyllama_experiment.npz",
    },
    "mistral_q4": {
        "model": "mlx_model_mistral_q4",
        "adapter": "outputs/adapters/adapters_pasalid_mistral_q4_experiment.npz",
    },
    "qwen3": {
        "model": "mlx-community/Qwen3-4B-8bit",
        "adapter": "outputs/adapters/adapters_pasalid_qwen3_experiment.npz",
    },
}


def run_eval(model: str, data_path: Path, adapter_file: str | None, lora_layers: int, max_new_tokens: int, preview: int) -> str:
    command = [
        "python3",
        "-m",
        "lora_mlx.evaluation",
        "--model",
        model,
        "--data",
        str(data_path),
        "--lora-layers",
        str(lora_layers),
        "--max-new-tokens",
        str(max_new_tokens),
        "--preview",
        str(preview),
    ]
    if adapter_file:
        command.extend(["--adapter-file", adapter_file])
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run A/B/C evaluation over Pasal.id experiment splits.")
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), default="tinyllama", help="Model preset")
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR), help="Experiment split directory")
    parser.add_argument("--split", choices=["seen", "unseen"], default="seen", help="Which evaluation split to use")
    parser.add_argument("--lora-layers", type=int, default=4, help="Number of LoRA layers")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Max generation tokens")
    parser.add_argument("--preview", type=int, default=5, help="Preview rows")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    preset = MODEL_PRESETS[args.preset]
    experiment_dir = Path(args.experiment_dir)

    if args.split == "seen":
        no_context = experiment_dir / "test_seen.jsonl"
        with_context = experiment_dir / "test_seen_with_context.jsonl"
    else:
        no_context = experiment_dir / "test_unseen.jsonl"
        with_context = experiment_dir / "test_unseen_with_context.jsonl"

    outputs = {
        "A_base_no_context": run_eval(
            preset["model"],
            no_context,
            adapter_file=None,
            lora_layers=args.lora_layers,
            max_new_tokens=args.max_new_tokens,
            preview=args.preview,
        ),
        "B_base_with_context": run_eval(
            preset["model"],
            with_context,
            adapter_file=None,
            lora_layers=args.lora_layers,
            max_new_tokens=args.max_new_tokens,
            preview=args.preview,
        ),
        "C_adapter_no_context": run_eval(
            preset["model"],
            no_context,
            adapter_file=preset["adapter"],
            lora_layers=args.lora_layers,
            max_new_tokens=args.max_new_tokens,
            preview=args.preview,
        ),
    }

    print(json.dumps({"preset": args.preset, "split": args.split, "outputs": outputs}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
