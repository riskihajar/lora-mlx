import argparse
import json
from pathlib import Path

from .evaluation import (
    exact_match,
    extract_parts,
    f1_score,
    generate_text,
    prepare_model,
)
from .paths import DEFAULT_DATA_DIR, DEFAULT_PREDICTIONS_DIR


def build_parser():
    parser = argparse.ArgumentParser(description="Export model predictions to JSONL.")
    parser.add_argument("--model", required=True, help="Path to local MLX model or HF repo")
    parser.add_argument(
        "--adapter-file",
        default=None,
        help="Optional LoRA adapter file to load before export",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_DIR / "test.jsonl"),
        help="Path to JSONL evaluation data",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=4,
        help="Number of last layers to wrap with LoRA when loading adapters",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of generated tokens per example",
    )
    return parser


def main():
    args = build_parser().parse_args()

    model, tokenizer = prepare_model(args.model, args.adapter_file, args.lora_layers)
    examples = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    output_path = Path(args.output)
    if not output_path.is_absolute() and output_path.parent == Path('.'):
        output_path = DEFAULT_PREDICTIONS_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as handle:
        for index, example in enumerate(examples, start=1):
            prompt, gold = extract_parts(example["text"])
            prediction = generate_text(model, tokenizer, prompt, args.max_new_tokens)
            row = {
                "index": index,
                "prompt": prompt,
                "gold": gold,
                "prediction": prediction,
                "em": exact_match(prediction, gold),
                "f1": round(f1_score(prediction, gold), 4),
            }
            handle.write(json.dumps(row) + "\n")

    print(f"wrote={len(examples)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
