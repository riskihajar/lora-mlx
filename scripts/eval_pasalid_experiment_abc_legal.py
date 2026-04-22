#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
from pathlib import Path

from lora_mlx.paths import DEFAULT_PREDICTIONS_DIR


DEFAULT_EXPORT_DIR = DEFAULT_PREDICTIONS_DIR / "pasalid_experiment"


def run_legal_eval(predictions_path: Path) -> dict:
    result = subprocess.run(
        [
            "python3",
            "scripts/eval_pasalid_legal_metrics.py",
            "--predictions",
            str(predictions_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    return json.loads(result.stdout)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate exported A/B/C Pasal.id predictions with legal-aware metrics.")
    parser.add_argument("--preset", required=True, help="Model preset name used in export filenames")
    parser.add_argument("--split", choices=["seen", "unseen"], default="seen", help="Which experiment split to evaluate")
    parser.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR), help="Prediction export directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_dir = Path(args.export_dir)
    labels = ["A_base_no_context", "B_base_with_context", "C_adapter_no_context"]
    outputs = {}
    for label in labels:
        path = export_dir / f"{args.preset}_{args.split}_{label}.jsonl"
        outputs[label] = run_legal_eval(path)
    print(json.dumps({"preset": args.preset, "split": args.split, "legal_metrics": outputs}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
