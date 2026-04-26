#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import mlx.core as mx

from lora_mlx import utils as lora_utils
from lora_mlx.doc_to_lora import GeneratedLoRA, apply_generated_loras


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query a model patched with generated Doc-to-LoRA weights."
    )
    parser.add_argument("--model", default="mlx-community/gemma-2-2b-it")
    parser.add_argument("--lora", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temp", type=float, default=0.0)
    return parser.parse_args()


def load_generated_loras(path: str) -> list[GeneratedLoRA]:
    lora_path = Path(path)
    metadata_path = lora_path.with_suffix(lora_path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text())
    weights = mx.load(str(lora_path))
    generated = []
    for idx, item in enumerate(metadata["loras"]):
        generated.append(
            GeneratedLoRA(
                layer_idx=int(item["layer_idx"]),
                module_name=item["module_name"],
                lora_a=weights[f"lora_{idx}_a"],
                lora_b=weights[f"lora_{idx}_b"],
                scale=float(item["scale"]),
            )
        )
    return generated


def generate_text(model, tokenizer, prompt: str, max_tokens: int, temp: float) -> str:
    prompt_ids = mx.array(tokenizer.encode(prompt), dtype=mx.int32)
    tokens = []
    for token, _ in zip(lora_utils.generate(prompt_ids, model, temp=temp), range(max_tokens)):
        token_id = int(token.item())
        if token_id == getattr(tokenizer, "eos_token_id", None):
            break
        tokens.append(token_id)
    return tokenizer.decode(tokens)


def main():
    args = parse_args()
    model, tokenizer, _ = lora_utils.load(args.model)
    model.freeze()
    generated_loras = load_generated_loras(args.lora)
    apply_generated_loras(model, generated_loras)
    mx.eval(model.parameters())
    text = generate_text(model, tokenizer, args.prompt, args.max_tokens, args.temp)
    print(args.prompt, end="")
    print(text)
    print(f"generated_tokens={len(tokenizer.encode(text))}")


if __name__ == "__main__":
    main()
