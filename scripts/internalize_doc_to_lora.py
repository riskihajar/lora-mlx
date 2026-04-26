#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import mlx.core as mx

from lora_mlx import utils as lora_utils
from lora_mlx.doc_to_lora import (
    DocToLoRAHypernetwork,
    infer_lora_module_specs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Internalize a document into generated Doc-to-LoRA weights."
    )
    parser.add_argument("--model", default="mlx-community/gemma-2-2b-it")
    parser.add_argument("--hypernet", required=True)
    parser.add_argument("--document", default=None)
    parser.add_argument("--document-file", default=None)
    parser.add_argument("--output", required=True)
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
    return parser.parse_args()


def read_document(args) -> str:
    if args.document is not None:
        return args.document
    if args.document_file is not None:
        return Path(args.document_file).read_text()
    raise ValueError("provide --document or --document-file")


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


def save_generated_loras(generated_loras, specs, args, document: str):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    metadata = {
        "model": args.model,
        "target_modules": args.target_modules,
        "rank": args.rank,
        "scale": 20.0,
        "context_encoder": "model-embed",
        "context_max_tokens": args.context_max_tokens,
        "context_chunk_tokens": args.context_chunk_tokens,
        "chunk_merge": args.chunk_merge,
        "document_chars": len(document),
        "loras": [],
    }
    for idx, generated in enumerate(generated_loras):
        arrays[f"lora_{idx}_a"] = generated.lora_a
        arrays[f"lora_{idx}_b"] = generated.lora_b
        metadata["loras"].append(
            {
                "layer_idx": generated.layer_idx,
                "module_name": generated.module_name,
                "scale": generated.scale,
                "input_dims": specs[idx].input_dims,
                "output_dims": specs[idx].output_dims,
            }
        )
    mx.savez(str(output), **arrays)
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2))
    print(f"saved_lora={output}")
    print(f"saved_metadata={output.with_suffix(output.suffix + '.json')}")
    print(f"num_loras={len(generated_loras)}")


def main():
    args = parse_args()
    document = read_document(args)
    model, tokenizer, _ = lora_utils.load(args.model)
    model.freeze()
    hypernet, specs = build_hypernet(model, args)
    document_ids = mx.array(tokenizer.encode(document), dtype=mx.int32)
    features = model_embedding_features(
        model,
        document_ids,
        args.context_max_tokens,
        args.context_chunk_tokens,
    )
    groups = [hypernet(feature) for feature in features]
    generated_loras = hypernet.merge_generated_lora_groups(groups)
    save_generated_loras(generated_loras, specs, args, document)


if __name__ == "__main__":
    main()
