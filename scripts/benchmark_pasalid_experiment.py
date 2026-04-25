#!/usr/bin/env python3

import argparse
import json
import re
import resource
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from lora_mlx.evaluation import (
    build_stop_token_sequences,
    extract_parts,
    generate_text,
    prepare_model,
)
from lora_mlx.paths import DEFAULT_DATA_DIR, DEFAULT_PREDICTIONS_DIR


DEFAULT_EXPERIMENT_DIR = DEFAULT_DATA_DIR / "pasalid" / "json_large_split"
DEFAULT_BENCHMARK_DIR = DEFAULT_PREDICTIONS_DIR / "pasalid_benchmarks"


MODEL_PRESETS = {
    "tinyllama": {
        "model": "mlx_model",
        "adapter": "outputs/adapters/adapters_pasalid_tinyllama_experiment.npz",
    },
    "tinyllama_native_expanded_clean": {
        "model": "mlx_model",
        "adapter": "outputs/adapters/adapters_pasalid_tinyllama_native_expanded_clean.npz",
    },
    "tinyllama_natural_legal": {
        "model": "mlx_model",
        "adapter": "outputs/adapters/adapters_pasalid_tinyllama_natural_legal.npz",
    },
    "mistral_q4": {
        "model": "mlx_model_mistral_q4",
        "adapter": "outputs/adapters/adapters_pasalid_mistral_q4_experiment_long.npz",
    },
    "qwen3": {
        "model": "mlx-community/Qwen3-4B-8bit",
        "adapter": "outputs/adapters/adapters_pasalid_qwen3_experiment.npz",
    },
}


def load_rows(path: Path, limit: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[:limit]


def prompt_token_proxy(text: str) -> int:
    return len(text.split())


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def run_export(model: str, data_path: Path, output_path: Path, adapter_file: str | None, lora_layers: int, max_new_tokens: int) -> tuple[float, int | None]:
    inner_command = [
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
        inner_command.extend(["--adapter-file", adapter_file])
    command = ["/usr/bin/time", "-l", *inner_command]
    start = time.perf_counter()
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    peak_rss = None
    match = re.search(r"(\d+)\s+maximum resident set size", result.stderr)
    if match:
        peak_rss = int(match.group(1))
    return elapsed, peak_rss


def current_peak_rss() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def benchmark_rows_per_example(
    model_path: str,
    rows: list[dict],
    output_path: Path,
    adapter_file: str | None,
    lora_layers: int,
    max_new_tokens: int,
) -> dict:
    model, tokenizer = prepare_model(model_path, adapter_file, lora_layers)
    stop_strings = ["\nQ:", "\nA:", "table:", "columns:"]
    stop_token_sequences = build_stop_token_sequences(tokenizer, stop_strings)
    token_counts = []
    generated_token_counts = []
    latencies = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as handle:
        for index, row in enumerate(rows, start=1):
            prompt, gold = extract_parts(row["text"])
            prompt_tokens = tokenizer.encode(prompt)
            token_counts.append(len(prompt_tokens))
            start = time.perf_counter()
            prediction = generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens,
                stop_strings,
                stop_token_sequences,
            )
            latency = time.perf_counter() - start
            generated_token_count = len(tokenizer.encode(prediction))
            generated_token_counts.append(generated_token_count)
            latencies.append(latency)
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "prompt_tokens": len(prompt_tokens),
                        "generated_tokens": generated_token_count,
                        "latency_seconds": latency,
                        "prompt": prompt,
                        "gold": gold,
                        "prediction": prediction,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    generated_total = sum(generated_token_counts)
    latency_total = sum(latencies)
    return {
        "avg_prompt_tokens": statistics.mean(token_counts) if token_counts else 0.0,
        "min_prompt_tokens": min(token_counts) if token_counts else 0,
        "max_prompt_tokens": max(token_counts) if token_counts else 0,
        "avg_generated_tokens": statistics.mean(generated_token_counts) if generated_token_counts else 0.0,
        "latency_total_seconds": latency_total,
        "latency_avg_seconds": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_seconds": percentile(latencies, 0.5),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "generated_tokens_per_second": generated_total / latency_total if latency_total else 0.0,
        "peak_rss_bytes_proxy": current_peak_rss(),
    }


def write_temp_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark A/B/C/D efficiency on Pasal.id experiment splits.")
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), default="tinyllama", help="Model preset")
    parser.add_argument("--split", choices=["seen", "unseen"], default="seen", help="Experiment split")
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR), help="Experiment split directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_BENCHMARK_DIR), help="Directory for benchmark outputs")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples per condition")
    parser.add_argument("--lora-layers", type=int, default=4, help="Number of LoRA layers")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max generation tokens")
    parser.add_argument("--per-example", action="store_true", help="Measure latency per example in-process and count tokenizer tokens")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    preset = MODEL_PRESETS[args.preset]
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "seen":
        no_context_path = experiment_dir / "test_seen.jsonl"
        with_context_path = experiment_dir / "test_seen_with_context.jsonl"
    else:
        no_context_path = experiment_dir / "test_unseen.jsonl"
        with_context_path = experiment_dir / "test_unseen_with_context.jsonl"

    a_rows = load_rows(no_context_path, args.limit)
    b_rows = load_rows(with_context_path, args.limit)
    c_rows = a_rows
    d_rows = b_rows

    summary = {
        "preset": args.preset,
        "split": args.split,
        "samples": args.limit,
        "per_example": args.per_example,
        "conditions": {},
    }

    conditions = {
        "A": {"rows": a_rows, "adapter": None},
        "B": {"rows": b_rows, "adapter": None},
        "C": {"rows": c_rows, "adapter": preset["adapter"]},
        "D": {"rows": d_rows, "adapter": preset["adapter"]},
    }

    if args.per_example:
        for label, cfg in conditions.items():
            output_path = output_dir / f"{args.preset}_{args.split}_{label}_per_example_benchmark_predictions.jsonl"
            summary["conditions"][label] = benchmark_rows_per_example(
                model_path=preset["model"],
                rows=cfg["rows"],
                output_path=output_path,
                adapter_file=cfg["adapter"],
                lora_layers=args.lora_layers,
                max_new_tokens=args.max_new_tokens,
            )
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            for label, cfg in conditions.items():
                data_path = tmp_dir_path / f"{label}.jsonl"
                output_path = output_dir / f"{args.preset}_{args.split}_{label}_benchmark_predictions.jsonl"
                write_temp_jsonl(cfg["rows"], data_path)
                token_counts = [prompt_token_proxy(row["text"]) for row in cfg["rows"]]
                latency, peak_rss = run_export(
                    model=preset["model"],
                    data_path=data_path,
                    output_path=output_path,
                    adapter_file=cfg["adapter"],
                    lora_layers=args.lora_layers,
                    max_new_tokens=args.max_new_tokens,
                )
                per_example = latency / len(cfg["rows"]) if cfg["rows"] else 0.0
                latencies = [per_example] * len(cfg["rows"])
                summary["conditions"][label] = {
                    "avg_prompt_token_proxy": statistics.mean(token_counts) if token_counts else 0.0,
                    "min_prompt_token_proxy": min(token_counts) if token_counts else 0,
                    "max_prompt_token_proxy": max(token_counts) if token_counts else 0,
                    "latency_total_seconds": latency,
                    "latency_avg_seconds": per_example,
                    "latency_p50_seconds": percentile(latencies, 0.5),
                    "latency_p95_seconds": percentile(latencies, 0.95),
                    "peak_rss_bytes_proxy": peak_rss,
                }

    summary_path = output_dir / f"{args.preset}_{args.split}_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
