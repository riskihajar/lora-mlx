import argparse
import json
import re
from collections import Counter
from pathlib import Path

import mlx.core as mx

from . import utils as lora_utils
from .models import LoRALinear
from .paths import DEFAULT_ADAPTERS_DIR, DEFAULT_DATA_DIR


def apply_lora_layers(model, lora_layers):
    for layer in model.model.layers[len(model.model.layers) - lora_layers :]:
        layer.self_attn.q_proj = LoRALinear.from_linear(layer.self_attn.q_proj)
        if layer.self_attn.v_proj is not None:
            layer.self_attn.v_proj = LoRALinear.from_linear(layer.self_attn.v_proj)
        if hasattr(layer, "block_sparse_moe"):
            layer.block_sparse_moe.gate = LoRALinear.from_linear(
                layer.block_sparse_moe.gate
            )


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate EM and F1 for WikiSQL-style data.")
    parser.add_argument("--model", required=True, help="Path to local MLX model or HF repo")
    parser.add_argument(
        "--adapter-file",
        default=None,
        help="Optional LoRA adapter file to load before evaluation",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_DIR / "test.jsonl"),
        help="Path to JSONL evaluation data",
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
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of examples to print as preview",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of examples to evaluate",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of examples to skip before evaluation",
    )
    parser.add_argument(
        "--stop-strings",
        nargs="*",
        default=["\nQ:", "\nA:", "table:", "columns:"],
        help="Stop generation early if decoded text contains any of these strings",
    )
    return parser


def extract_parts(text):
    if "\nA: " not in text:
        return None, None
    prompt, answer = text.split("\nA: ", 1)
    return prompt + "\nA: ", answer.strip()


def prepare_model(model_path, adapter_file=None, lora_layers=4):
    model, tokenizer, _ = lora_utils.load(model_path)
    if adapter_file is not None:
        model.freeze()
        apply_lora_layers(model, lora_layers)
        model.load_weights(adapter_file, strict=False)
    model.eval()
    return model, tokenizer


def build_stop_token_sequences(tokenizer, stop_strings):
    sequences = []
    for stop_string in stop_strings or []:
        token_ids = tokenizer.encode(stop_string)
        if token_ids:
            sequences.append((stop_string, token_ids))
    return sequences


def trim_stop_sequences(text, stop_strings):
    for stop_string in stop_strings or []:
        stop_index = text.find(stop_string)
        if stop_index != -1:
            return text[:stop_index].strip()
    return text.strip()


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=64,
    stop_strings=None,
    stop_token_sequences=None,
):
    prompt_ids = mx.array(tokenizer.encode(prompt))
    tokens = []
    stop_strings = stop_strings or []
    stop_token_sequences = stop_token_sequences or []
    for token, _ in zip(
        lora_utils.generate(prompt_ids, model, temp=0.0),
        range(max_new_tokens),
    ):
        token_id = token.item()
        if token_id == tokenizer.eos_token_id:
            break
        tokens.append(token_id)
        for _, stop_token_ids in stop_token_sequences:
            if len(tokens) >= len(stop_token_ids) and tokens[-len(stop_token_ids) :] == stop_token_ids:
                decoded = tokenizer.decode(tokens[:-len(stop_token_ids)])
                return decoded.strip()
    decoded = tokenizer.decode(tokens)
    return trim_stop_sequences(decoded, stop_strings)


def normalize_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_sql_tokens(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([(),=])\s*", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().split()


def exact_match(prediction, gold):
    return int(normalize_text(prediction) == normalize_text(gold))


def f1_score(prediction, gold):
    pred_tokens = normalize_sql_tokens(prediction)
    gold_tokens = normalize_sql_tokens(gold)
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


def main():
    args = build_parser().parse_args()

    model, tokenizer = prepare_model(args.model, args.adapter_file, args.lora_layers)
    examples = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    stop_token_sequences = build_stop_token_sequences(tokenizer, args.stop_strings)
    if args.offset:
        examples = examples[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("No evaluation examples selected. Check --offset/--limit.")

    ems = []
    f1s = []
    previews = []

    for idx, example in enumerate(examples):
        prompt, gold = extract_parts(example["text"])
        prediction = generate_text(
            model,
            tokenizer,
            prompt,
            args.max_new_tokens,
            args.stop_strings,
            stop_token_sequences,
        )
        em = exact_match(prediction, gold)
        f1 = f1_score(prediction, gold)
        ems.append(em)
        f1s.append(f1)
        if idx < args.preview:
            previews.append((idx + 1, prediction, gold, em, f1))

    print(f"examples={len(examples)}")
    print(f"em={sum(ems) / len(ems):.4f}")
    print(f"f1={sum(f1s) / len(f1s):.4f}")
    print("preview_start")
    for idx, prediction, gold, em, f1 in previews:
        print(f"[{idx}] EM={em} F1={f1:.4f}")
        print(f"PRED: {prediction}")
        print(f"GOLD: {gold}")
    print("preview_end")


if __name__ == "__main__":
    main()
