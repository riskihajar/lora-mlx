#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from lora_mlx import utils as lora_utils
from lora_mlx.doc_to_lora import (
    DocToLoRAHypernetwork,
    GeneratedLoRALinear,
    infer_lora_module_specs,
)


@dataclass(frozen=True)
class EvalExample:
    document: str
    document_ids: mx.array
    prompt_ids: mx.array
    response_ids: mx.array


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate no-source-context Doc-to-LoRA internalization on JSONL."
    )
    parser.add_argument("--model", default="mlx-community/gemma-2-2b-it")
    parser.add_argument("--hypernet", required=True)
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--skip-examples", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lora-layers", type=int, default=2)
    parser.add_argument("--target-modules", default="down_proj")
    parser.add_argument("--max-specs", type=int, default=2)
    parser.add_argument("--context-max-tokens", type=int, default=512)
    parser.add_argument("--context-chunk-tokens", type=int, default=128)
    parser.add_argument("--chunk-merge", choices=["mean", "learned"], default="learned")
    parser.add_argument("--max-context-chunks", type=int, default=8)
    parser.add_argument("--per-rank-gen", action="store_true", default=True)
    parser.add_argument("--per-layer-processing", action="store_true", default=True)
    parser.add_argument("--num-pre-head-layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def load_examples(
    tokenizer,
    path: str,
    skip_examples: int,
    max_examples: int,
) -> list[EvalExample]:
    examples = []
    seen = 0
    with open(path, "r") as fid:
        for line in fid:
            if not line.strip():
                continue
            if seen < skip_examples:
                seen += 1
                continue
            seen += 1
            row = json.loads(line)
            document = row.get("document") or row.get("context")
            prompt = row.get("prompt") or row.get("question")
            response = row.get("response") or row.get("answer")
            if not document or not prompt or not response:
                continue
            prompt_ids = row.get("prompt_ids") or tokenizer.encode(prompt)
            response_ids = row.get("response_ids") or tokenizer.encode(response)
            if not prompt_ids or not response_ids:
                continue
            examples.append(
                EvalExample(
                    document=document,
                    document_ids=mx.array(tokenizer.encode(document), dtype=mx.int32),
                    prompt_ids=mx.array(prompt_ids, dtype=mx.int32),
                    response_ids=mx.array(response_ids, dtype=mx.int32),
                )
            )
            if len(examples) >= max_examples:
                break
    if not examples:
        raise ValueError(f"no usable examples found in {path}")
    return examples


def chunk_context_ids(document_ids: mx.array, max_context_tokens: int, chunk_tokens: int):
    if max_context_tokens > 0 and len(document_ids) > max_context_tokens:
        document_ids = document_ids[:max_context_tokens]
    if chunk_tokens <= 0:
        return [document_ids]
    return [
        document_ids[start : start + chunk_tokens]
        for start in range(0, len(document_ids), chunk_tokens)
    ]


def model_embedding_features(model, document_ids, max_context_tokens: int, chunk_tokens: int):
    embed = model.model.embed_tokens
    scale = getattr(model.model, "embed_scale", 1.0)
    features = []
    for context_ids in chunk_context_ids(document_ids, max_context_tokens, chunk_tokens):
        embeddings = embed(context_ids[None, :])[0] * scale
        features.append(mx.mean(embeddings, axis=0))
    mx.eval(features)
    return features


def build_hypernet(model, args):
    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    specs = infer_lora_module_specs(
        model,
        target_modules=target_modules,
        lora_layers=args.lora_layers,
    )
    if args.max_specs > 0:
        specs = specs[: args.max_specs]
    if not specs:
        raise ValueError("no target LoRA modules found")
    hypernet = DocToLoRAHypernetwork(
        specs,
        feature_size=model.args.hidden_size,
        hidden_size=args.hidden_size,
        rank=args.rank,
        scale=20.0,
        per_rank_gen=args.per_rank_gen,
        per_layer_processing=args.per_layer_processing,
        num_pre_head_layers=args.num_pre_head_layers,
        chunk_merge=args.chunk_merge,
        max_context_chunks=args.max_context_chunks,
    )
    hypernet.load_weights(args.hypernet, strict=False)
    mx.eval(hypernet.parameters())
    return hypernet, specs


def generated_loras_for_example(model, hypernet, example: EvalExample, args):
    features = model_embedding_features(
        model,
        example.document_ids,
        args.context_max_tokens,
        args.context_chunk_tokens,
    )
    groups = [hypernet(feature) for feature in features]
    return hypernet.merge_generated_lora_groups(groups)


def apply_dynamic_generated_loras(model, generated_loras):
    layers = model.model.layers
    for generated_lora in generated_loras:
        layer = layers[generated_lora.layer_idx]
        if generated_lora.module_name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            current = getattr(layer.self_attn, generated_lora.module_name)
            linear = current.linear if isinstance(current, GeneratedLoRALinear) else current
            setattr(
                layer.self_attn,
                generated_lora.module_name,
                GeneratedLoRALinear(linear, generated_lora),
            )
        elif generated_lora.module_name in {"gate_proj", "down_proj", "up_proj"}:
            current = getattr(layer.mlp, generated_lora.module_name)
            linear = current.linear if isinstance(current, GeneratedLoRALinear) else current
            setattr(
                layer.mlp,
                generated_lora.module_name,
                GeneratedLoRALinear(linear, generated_lora),
            )
        else:
            raise ValueError(f"unsupported module {generated_lora.module_name!r}")


def teacher_forced_logits(model, example: EvalExample):
    response_ids = example.response_ids
    if len(response_ids) > 1:
        input_ids = mx.concatenate([example.prompt_ids, response_ids[:-1]], axis=0)
    else:
        input_ids = example.prompt_ids
    logits, _ = model(input_ids[None, :])
    start = len(example.prompt_ids) - 1
    end = start + len(response_ids)
    return logits[0, start:end, :].astype(mx.float32), response_ids.astype(mx.int32)


def evaluate(model, hypernet, examples: list[EvalExample], args):
    losses = []
    correct = 0
    total = 0
    exact = 0
    for example in examples:
        generated_loras = generated_loras_for_example(model, hypernet, example, args)
        apply_dynamic_generated_loras(model, generated_loras)
        logits, targets = teacher_forced_logits(model, example)
        losses.append(nn.losses.cross_entropy(logits, targets).mean())
        preds = mx.argmax(logits, axis=-1)
        correct += int(mx.sum(preds == targets).item())
        total += len(targets)
        exact += int(mx.all(preds == targets).item())
    loss = mx.mean(mx.stack(losses)).item()
    return {
        "loss": loss,
        "token_acc": correct / total if total else 0.0,
        "exact_acc": exact / len(examples),
        "response_tokens": total,
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    model, tokenizer, _ = lora_utils.load(args.model)
    model.freeze()
    hypernet, specs = build_hypernet(model, args)
    examples = load_examples(
        tokenizer,
        args.dataset_jsonl,
        args.skip_examples,
        args.max_examples,
    )
    metrics = evaluate(model, hypernet, examples, args)
    print(f"examples={len(examples)}")
    print(f"skip_examples={args.skip_examples}")
    print(f"response_tokens={metrics['response_tokens']}")
    print(f"internalized_loss={metrics['loss']:.6f}")
    print(f"internalized_token_acc={metrics['token_acc']:.3f}")
    print(f"internalized_exact_acc={metrics['exact_acc']:.3f}")
    print(f"hypernet={args.hypernet}")
    print(f"num_specs={len(specs)}")
    print(f"context_chunk_tokens={args.context_chunk_tokens}")
    print(f"chunk_merge={args.chunk_merge}")


if __name__ == "__main__":
    main()
