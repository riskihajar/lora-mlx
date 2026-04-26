#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from lora_mlx import utils as lora_utils
from lora_mlx.doc_to_lora import (
    DocToLoRAHypernetwork,
    GeneratedLoRALinear,
    infer_lora_module_specs,
)
from lora_mlx.models import LoRALinear


@dataclass(frozen=True)
class GenerationExample:
    document: str
    document_ids: mx.array
    prompt: str
    prompt_ids: mx.array
    response: str
    response_ids: mx.array


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate answers for Sakana JSONL examples and score exact/F1."
    )
    parser.add_argument(
        "--mode",
        choices=["base", "generated", "ordinary"],
        required=True,
        help="Model variant to evaluate. Run once per mode to keep memory usage low.",
    )
    parser.add_argument("--model", default="mlx-community/gemma-2-2b-it")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--skip-examples", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument(
        "--suppress-initial-whitespace",
        action="store_true",
        help="When greedy decoding, skip whitespace-only candidates until content appears.",
    )
    parser.add_argument(
        "--top-k-fallback",
        type=int,
        default=32,
        help="Number of candidates to scan when suppressing whitespace-only tokens.",
    )
    parser.add_argument(
        "--stop-on-end-turn",
        action="store_true",
        help="Stop decoding once the decoded output contains <end_of_turn>.",
    )
    parser.add_argument("--hypernet", default=None)
    parser.add_argument("--ordinary-adapter", default=None)
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
    parser.add_argument(
        "--show-examples",
        type=int,
        default=3,
        help="Print the first N generations for qualitative inspection.",
    )
    return parser.parse_args()


def apply_hypernet_config(args):
    if not args.hypernet:
        return args
    config_path = Path(args.hypernet).with_suffix(Path(args.hypernet).suffix + ".json")
    if not config_path.exists():
        return args
    config = json.loads(config_path.read_text())
    for key in [
        "hidden_size",
        "rank",
        "lora_layers",
        "target_modules",
        "max_specs",
        "context_max_tokens",
        "context_chunk_tokens",
        "chunk_merge",
        "max_context_chunks",
        "per_rank_gen",
        "per_layer_processing",
        "num_pre_head_layers",
        "seed",
    ]:
        if key in config:
            setattr(args, key, config[key])
    print(f"loaded_hypernet_config={config_path}")
    return args


def load_examples(tokenizer, path: str, skip_examples: int, max_examples: int):
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
                GenerationExample(
                    document=document,
                    document_ids=mx.array(tokenizer.encode(document), dtype=mx.int32),
                    prompt=prompt,
                    prompt_ids=mx.array(prompt_ids, dtype=mx.int32),
                    response=response,
                    response_ids=mx.array(response_ids, dtype=mx.int32),
                )
            )
            if len(examples) >= max_examples:
                break
    if not examples:
        raise ValueError(f"no usable examples found in {path}")
    return examples


def target_specs(model, args):
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
    return specs


def build_hypernet(model, args, specs):
    if not args.hypernet:
        raise ValueError("--mode generated requires --hypernet")
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
    return hypernet


def apply_ordinary_lora(model, args, specs):
    if not args.ordinary_adapter:
        raise ValueError("--mode ordinary requires --ordinary-adapter")
    for spec in specs:
        layer = model.model.layers[spec.layer_idx]
        if spec.module_name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            current = getattr(layer.self_attn, spec.module_name)
            setattr(layer.self_attn, spec.module_name, LoRALinear.from_linear(current, rank=args.rank))
        elif spec.module_name in {"gate_proj", "down_proj", "up_proj"}:
            current = getattr(layer.mlp, spec.module_name)
            setattr(layer.mlp, spec.module_name, LoRALinear.from_linear(current, rank=args.rank))
        else:
            raise ValueError(f"unsupported module {spec.module_name!r}")
    model.load_weights(args.ordinary_adapter, strict=False)
    mx.eval(model.parameters())


def chunk_context_ids(document_ids: mx.array, max_context_tokens: int, chunk_tokens: int):
    if max_context_tokens > 0 and len(document_ids) > max_context_tokens:
        document_ids = document_ids[:max_context_tokens]
    if chunk_tokens <= 0:
        return [document_ids]
    return [
        document_ids[start : start + chunk_tokens]
        for start in range(0, len(document_ids), chunk_tokens)
    ]


def generated_loras_for_example(model, hypernet, example: GenerationExample, args):
    embed = model.model.embed_tokens
    scale = getattr(model.model, "embed_scale", 1.0)
    groups = []
    for context_ids in chunk_context_ids(
        example.document_ids,
        args.context_max_tokens,
        args.context_chunk_tokens,
    ):
        embeddings = embed(context_ids[None, :])[0] * scale
        feature = mx.mean(embeddings, axis=0)
        groups.append(hypernet(feature))
    return hypernet.merge_generated_lora_groups(groups)


def apply_dynamic_generated_loras(model, generated_loras):
    layers = model.model.layers
    for generated_lora in generated_loras:
        layer = layers[generated_lora.layer_idx]
        if generated_lora.module_name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            current = getattr(layer.self_attn, generated_lora.module_name)
            linear = current.linear if isinstance(current, GeneratedLoRALinear) else current
            setattr(layer.self_attn, generated_lora.module_name, GeneratedLoRALinear(linear, generated_lora))
        elif generated_lora.module_name in {"gate_proj", "down_proj", "up_proj"}:
            current = getattr(layer.mlp, generated_lora.module_name)
            linear = current.linear if isinstance(current, GeneratedLoRALinear) else current
            setattr(layer.mlp, generated_lora.module_name, GeneratedLoRALinear(linear, generated_lora))
        else:
            raise ValueError(f"unsupported module {generated_lora.module_name!r}")


def sample_token(
    logits: mx.array,
    tokenizer,
    temp: float,
    suppress_whitespace: bool,
    top_k_fallback: int,
):
    if temp != 0:
        return int(mx.random.categorical(logits * (1 / temp)).item())
    if not suppress_whitespace:
        return int(mx.argmax(logits, axis=-1).item())
    scores = np.array(logits, copy=False)
    candidate_count = min(max(top_k_fallback, 1), scores.shape[-1])
    candidate_ids = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
    candidate_ids = candidate_ids[np.argsort(-scores[candidate_ids])]
    for token_id in candidate_ids:
        piece = tokenizer.decode([int(token_id)])
        if "<end_of_turn>" in piece or "<start_of_turn>" in piece:
            continue
        if piece.strip():
            return int(token_id)
    return int(candidate_ids[0])


def generate_ids(
    model,
    prompt_ids: mx.array,
    tokenizer,
    max_new_tokens: int,
    temp: float,
    eos_token_id,
    suppress_initial_whitespace: bool,
    top_k_fallback: int,
    stop_on_end_turn: bool,
):
    tokens = []
    y = prompt_ids
    cache = None
    emitted_content = False
    for _ in range(max_new_tokens):
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :][0]
        token_id = sample_token(
            logits,
            tokenizer,
            temp,
            suppress_initial_whitespace and not emitted_content,
            top_k_fallback,
        )
        if eos_token_id is not None and token_id == eos_token_id:
            break
        tokens.append(token_id)
        emitted_content = emitted_content or bool(tokenizer.decode([token_id]).strip())
        if stop_on_end_turn and "<end_of_turn>" in tokenizer.decode(tokens):
            break
        y = mx.array([token_id], dtype=mx.int32)
        mx.eval(y)
    return tokens


def normalize(text: str):
    text = text.replace("<end_of_turn>", " ").replace("<start_of_turn>", " ")
    return re.findall(r"\w+", text.lower())


def token_f1(prediction: str, reference: str):
    pred_tokens = normalize(prediction)
    ref_tokens = normalize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str):
    return int(" ".join(normalize(prediction)) == " ".join(normalize(reference)))


def main():
    args = apply_hypernet_config(parse_args())
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    model, tokenizer, _ = lora_utils.load(args.model)
    model.freeze()
    specs = target_specs(model, args)
    hypernet = None
    if args.mode == "generated":
        hypernet = build_hypernet(model, args, specs)
    elif args.mode == "ordinary":
        apply_ordinary_lora(model, args, specs)
    else:
        mx.eval(model.parameters())

    examples = load_examples(tokenizer, args.dataset_jsonl, args.skip_examples, args.max_examples)
    f1_scores = []
    exact_scores = []
    output_lengths = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    for idx, example in enumerate(examples):
        if hypernet is not None:
            generated_loras = generated_loras_for_example(model, hypernet, example, args)
            apply_dynamic_generated_loras(model, generated_loras)
        output_ids = generate_ids(
            model,
            example.prompt_ids,
            tokenizer,
            args.max_new_tokens,
            args.temp,
            eos_token_id,
            args.suppress_initial_whitespace,
            args.top_k_fallback,
            args.stop_on_end_turn,
        )
        prediction = tokenizer.decode(output_ids)
        reference = tokenizer.decode(example.response_ids.tolist())
        f1 = token_f1(prediction, reference)
        exact = exact_match(prediction, reference)
        f1_scores.append(f1)
        exact_scores.append(exact)
        output_lengths.append(len(output_ids))
        if idx < args.show_examples:
            print(f"example_{idx}_reference={reference!r}")
            print(f"example_{idx}_prediction={prediction!r}")
            print(f"example_{idx}_f1={f1:.3f}")
            print(f"example_{idx}_exact={exact}")

    print(f"mode={args.mode}")
    print(f"examples={len(examples)}")
    print(f"skip_examples={args.skip_examples}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print(f"suppress_initial_whitespace={args.suppress_initial_whitespace}")
    print(f"top_k_fallback={args.top_k_fallback}")
    print(f"stop_on_end_turn={args.stop_on_end_turn}")
    print(f"mean_f1={float(np.mean(f1_scores)):.6f}")
    print(f"exact_acc={float(np.mean(exact_scores)):.6f}")
    print(f"mean_generated_tokens={float(np.mean(output_lengths)):.2f}")
    print(f"target_modules={args.target_modules}")
    print(f"num_specs={len(specs)}")
    if args.hypernet:
        print(f"hypernet={args.hypernet}")
    if args.ordinary_adapter:
        print(f"ordinary_adapter={args.ordinary_adapter}")


if __name__ == "__main__":
    main()
