#!/usr/bin/env python3
import argparse
import json
import math
import re
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
from lora_mlx.models import LoRALinear


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "been",
    "by",
    "for",
    "from",
    "had",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "see",
    "some",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "where",
    "with",
}


@dataclass(frozen=True)
class RescoringExample:
    document: str
    document_ids: mx.array
    prompt_ids: mx.array
    response: str
    response_ids: mx.array


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank answer candidates by model likelihood on Sakana JSONL examples."
    )
    parser.add_argument("--mode", choices=["base", "generated", "ordinary"], required=True)
    parser.add_argument("--model", default="mlx-community/gemma-2-2b-it")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--skip-examples", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=16)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument(
        "--candidate-source",
        choices=["responses", "doc-spans", "mixed"],
        default="responses",
        help="Use held-out responses, document spans, or both as distractor candidates.",
    )
    parser.add_argument(
        "--allow-stopword-spans",
        action="store_true",
        help="Keep single-token stopword document spans as candidates.",
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
    parser.add_argument("--show-examples", type=int, default=3)
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
                RescoringExample(
                    document=document,
                    document_ids=mx.array(tokenizer.encode(document), dtype=mx.int32),
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
    specs = infer_lora_module_specs(model, target_modules=target_modules, lora_layers=args.lora_layers)
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
    return [document_ids[start : start + chunk_tokens] for start in range(0, len(document_ids), chunk_tokens)]


def generated_loras_for_example(model, hypernet, example: RescoringExample, args):
    embed = model.model.embed_tokens
    scale = getattr(model.model, "embed_scale", 1.0)
    groups = []
    for context_ids in chunk_context_ids(example.document_ids, args.context_max_tokens, args.context_chunk_tokens):
        embeddings = embed(context_ids[None, :])[0] * scale
        groups.append(hypernet(mx.mean(embeddings, axis=0)))
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


def candidate_ids(tokenizer, text: str):
    ids = tokenizer.encode(text)
    if not ids:
        ids = [getattr(tokenizer, "eos_token_id", 1) or 1]
    return mx.array(ids, dtype=mx.int32)


def candidate_loss(model, prompt_ids: mx.array, response_ids: mx.array):
    if len(response_ids) > 1:
        input_ids = mx.concatenate([prompt_ids, response_ids[:-1]], axis=0)
    else:
        input_ids = prompt_ids
    logits, _ = model(input_ids[None, :])
    start = len(prompt_ids) - 1
    end = start + len(response_ids)
    logits = logits[0, start:end, :].astype(mx.float32)
    return nn.losses.cross_entropy(logits, response_ids.astype(mx.int32)).mean().item()


def normalize_answer(text: str):
    return " ".join(re.findall(r"\w+", text.lower()))


def add_unique(candidates: list[str], seen: set[str], candidate: str):
    candidate = candidate.strip()
    key = normalize_answer(candidate)
    if not candidate or not key or key in seen:
        return
    candidates.append(candidate)
    seen.add(key)


def document_span_candidates(
    example: RescoringExample,
    max_candidates: int,
    allow_stopword_spans: bool,
):
    cleaned = example.document.replace("user", " ").replace("model", " ")
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", cleaned)
    gold_len = max(1, len(normalize_answer(example.response).split()))
    ngram_lengths = sorted({gold_len, max(1, gold_len - 1), gold_len + 1})
    candidates = []
    seen = set()
    add_unique(candidates, seen, example.response)
    for ngram_len in ngram_lengths:
        for start in range(0, max(len(words) - ngram_len + 1, 0)):
            text = " ".join(words[start : start + ngram_len])
            if len(text) < 3:
                continue
            if (
                not allow_stopword_spans
                and ngram_len == 1
                and text.lower() in STOPWORDS
            ):
                continue
            add_unique(candidates, seen, text)
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def response_candidates(examples: list[RescoringExample], idx: int, max_candidates: int):
    all_responses = []
    for example in examples:
        if example.response not in all_responses:
            all_responses.append(example.response)
    example = examples[idx]
    candidates = []
    seen = set()
    add_unique(candidates, seen, example.response)
    cursor = 1
    while len(candidates) < min(max_candidates, len(all_responses)):
        candidate = all_responses[(idx + cursor) % len(all_responses)]
        cursor += 1
        add_unique(candidates, seen, candidate)
    return candidates


def build_candidate_sets(
    examples: list[RescoringExample],
    num_candidates: int,
    source: str,
    allow_stopword_spans: bool,
):
    candidate_sets = []
    for idx, example in enumerate(examples):
        if source == "responses":
            candidates = response_candidates(examples, idx, num_candidates)
        elif source == "doc-spans":
            candidates = document_span_candidates(example, num_candidates, allow_stopword_spans)
        else:
            candidates = []
            seen = set()
            for candidate in document_span_candidates(
                example,
                num_candidates,
                allow_stopword_spans,
            ):
                add_unique(candidates, seen, candidate)
            for candidate in response_candidates(examples, idx, num_candidates):
                add_unique(candidates, seen, candidate)
                if len(candidates) >= num_candidates:
                    break
        candidate_sets.append(candidates)
    return candidate_sets


def evaluate(model, tokenizer, hypernet, examples, candidate_sets, args):
    ranks = []
    margins = []
    for idx, (example, candidates) in enumerate(zip(examples, candidate_sets)):
        if hypernet is not None:
            generated_loras = generated_loras_for_example(model, hypernet, example, args)
            apply_dynamic_generated_loras(model, generated_loras)
        losses = [candidate_loss(model, example.prompt_ids, candidate_ids(tokenizer, text)) for text in candidates]
        order = np.argsort(losses)
        rank = int(np.where(order == 0)[0][0]) + 1
        best_wrong = min(losses[1:]) if len(losses) > 1 else math.inf
        margin = best_wrong - losses[0]
        ranks.append(rank)
        margins.append(margin)
        if idx < args.show_examples:
            print(f"example_{idx}_gold={candidates[0]!r}")
            print(f"example_{idx}_gold_loss={losses[0]:.6f}")
            print(f"example_{idx}_best={candidates[int(order[0])]!r}")
            print(f"example_{idx}_best_loss={losses[int(order[0])]:.6f}")
            print(f"example_{idx}_rank={rank}")
            print(f"example_{idx}_margin={margin:.6f}")
    ranks_arr = np.array(ranks, dtype=np.float32)
    return {
        "top1_acc": float(np.mean(ranks_arr == 1)),
        "mean_rank": float(np.mean(ranks_arr)),
        "mrr": float(np.mean(1.0 / ranks_arr)),
        "mean_margin": float(np.mean(margins)),
    }


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
    candidate_sets = build_candidate_sets(
        examples,
        args.num_candidates,
        args.candidate_source,
        args.allow_stopword_spans,
    )
    metrics = evaluate(model, tokenizer, hypernet, examples, candidate_sets, args)
    print(f"mode={args.mode}")
    print(f"examples={len(examples)}")
    print(f"skip_examples={args.skip_examples}")
    print(f"num_candidates={min(args.num_candidates, len(examples))}")
    print(f"candidate_source={args.candidate_source}")
    print(f"allow_stopword_spans={args.allow_stopword_spans}")
    print(f"top1_acc={metrics['top1_acc']:.6f}")
    print(f"mean_rank={metrics['mean_rank']:.6f}")
    print(f"mrr={metrics['mrr']:.6f}")
    print(f"mean_margin={metrics['mean_margin']:.6f}")
    print(f"target_modules={args.target_modules}")
    print(f"num_specs={len(specs)}")
    if args.hypernet:
        print(f"hypernet={args.hypernet}")
    if args.ordinary_adapter:
        print(f"ordinary_adapter={args.ordinary_adapter}")


if __name__ == "__main__":
    main()
