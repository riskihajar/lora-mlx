#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from lora_mlx import utils as lora_utils
from lora_mlx.doc_to_lora import (
    DocToLoRAHypernetwork,
    GeneratedLoRALinear,
    PerceiverDocToLoRAHypernetwork,
    TokenDocToLoRAHypernetwork,
    infer_lora_module_specs,
    merge_generated_lora_groups,
)
from lora_mlx.models import Model, ModelArgs


@dataclass(frozen=True)
class TokenExample:
    document: str
    document_ids: mx.array
    prompt_ids: mx.array
    response_ids: mx.array
    document_features: mx.array | list[mx.array] | None = None
    logprobs_vals: mx.array | None = None
    logprobs_indices: mx.array | None = None


class ToyTokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def encode(self, text: str):
        values = []
        for token in text.lower().split():
            bucket = sum(token.encode("utf-8")) % (self.vocab_size - 4)
            values.append(bucket + 4)
        return values or [1]

    def convert_tokens_to_ids(self, token: str):
        return sum(token.encode("utf-8")) % self.vocab_size


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a native MLX Doc-to-LoRA hypernetwork through token loss."
    )
    parser.add_argument("--model", default="mlx_model")
    parser.add_argument("--toy", action="store_true", help="Use a tiny random MLX model.")
    parser.add_argument("--num-docs", type=int, default=4)
    parser.add_argument("--feature-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--context-buckets", type=int, default=4096)
    parser.add_argument("--context-latents", type=int, default=1)
    parser.add_argument(
        "--hypernet-aggregator",
        choices=["mlp", "perceiver"],
        default="mlp",
        help="Use the legacy MLP generator or a Sakana-style Perceiver bottleneck.",
    )
    parser.add_argument("--perceiver-latents", type=int, default=64)
    parser.add_argument("--perceiver-blocks", type=int, default=1)
    parser.add_argument("--perceiver-self-attn", type=int, default=1)
    parser.add_argument(
        "--spec-conditioning",
        action="store_true",
        help="Add learned layer/module embeddings before each LoRA head.",
    )
    parser.add_argument(
        "--per-rank-gen",
        action="store_true",
        help="Generate each LoRA rank from a separate rank-conditioned latent.",
    )
    parser.add_argument(
        "--per-layer-processing",
        action="store_true",
        help="Add layer embeddings and residual MLP blocks before LoRA heads.",
    )
    parser.add_argument("--num-pre-head-layers", type=int, default=1)
    parser.add_argument(
        "--context-encoder",
        choices=["hash", "token-hash", "model-embed", "model-activations"],
        default="hash",
        help="Use text hash, trainable token hash, frozen embeddings, or frozen layer activations.",
    )
    parser.add_argument("--context-max-tokens", type=int, default=1024)
    parser.add_argument(
        "--context-chunk-tokens",
        type=int,
        default=0,
        help="Split model-derived context into chunks and merge generated LoRAs.",
    )
    parser.add_argument(
        "--chunk-merge",
        choices=["mean", "learned"],
        default="mean",
        help="Merge chunk-generated LoRAs by averaging or learned per-target weights.",
    )
    parser.add_argument("--max-context-chunks", type=int, default=8)
    parser.add_argument(
        "--activation-pooling",
        choices=["mean", "latent"],
        default="mean",
        help="Pool frozen layer activations by token mean or trainable latent attention.",
    )
    parser.add_argument("--activation-latents", type=int, default=4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lora-layers", type=int, default=1)
    parser.add_argument("--target-modules", default="down_proj")
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Optional number of train examples per optimizer step; 0 uses all train examples.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--toy-vocab-size", type=int, default=96)
    parser.add_argument("--max-specs", type=int, default=2)
    parser.add_argument(
        "--dataset-jsonl",
        default=None,
        help="Optional JSONL with document/context, prompt/question, and response/answer fields.",
    )
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument(
        "--eval-examples",
        type=int,
        default=0,
        help="Hold out the last N loaded examples for eval-only metrics.",
    )
    parser.add_argument(
        "--loss-scope",
        choices=["first-token", "full-answer"],
        default="full-answer",
        help="Train on only the first response token or all response tokens.",
    )
    parser.add_argument(
        "--loss-type",
        choices=["ce", "kl-topk"],
        default="ce",
        help="Use hard-token cross entropy or sparse top-k teacher logprob loss.",
    )
    parser.add_argument(
        "--target-token-prefix",
        default="the",
        help="Tokenizer token used as the first synthetic target; later docs offset from it.",
    )
    parser.add_argument(
        "--save-hypernet",
        default=None,
        help="Optional path to save final hypernetwork weights as an MLX npz checkpoint.",
    )
    parser.add_argument(
        "--save-best-hypernet",
        default=None,
        help="Optional path to save the best hypernetwork checkpoint by eval loss.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="When saving best checkpoints, evaluate every N training steps.",
    )
    parser.add_argument(
        "--load-hypernet",
        default=None,
        help="Optional path to load hypernetwork weights before training/eval.",
    )
    parser.add_argument(
        "--save-optimizer",
        default=None,
        help="Optional path to save optimizer state as an MLX npz checkpoint.",
    )
    parser.add_argument(
        "--load-optimizer",
        default=None,
        help="Optional path to load optimizer state before training.",
    )
    return parser.parse_args()


def build_toy_model(args):
    model_args = ModelArgs(
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-5,
        vocab_size=args.toy_vocab_size,
    )
    model = Model(model_args)
    tokenizer = ToyTokenizer(args.toy_vocab_size)
    return model, tokenizer


def load_model(args):
    if args.toy:
        return build_toy_model(args)
    model_path = Path(args.model)
    if not model_path.exists() and args.model == "mlx_model":
        outputs_model = Path("outputs/models/mlx_model")
        if outputs_model.exists():
            args.model = str(outputs_model)
    return lora_utils.load(args.model)[:2]


def build_examples(tokenizer, args) -> list[TokenExample]:
    if args.dataset_jsonl:
        return build_examples_from_jsonl(tokenizer, args.dataset_jsonl, args.max_examples)

    examples = []
    vocab_size = get_vocab_size(tokenizer)
    base_target_id = get_token_id(tokenizer, args.target_token_prefix, default=8)
    for doc_id in range(args.num_docs):
        target_id = (base_target_id + doc_id) % vocab_size
        document = (
            f"Synthetic document {doc_id}. "
            f"The internalized answer token id is {target_id}."
        )
        prompt = f"Document memory {doc_id}. Question secret answer? Answer"
        prompt_ids = tokenizer.encode(prompt)
        examples.append(
            TokenExample(
                document=document,
                document_ids=mx.array(tokenizer.encode(document), dtype=mx.int32),
                prompt_ids=mx.array(prompt_ids, dtype=mx.int32),
                response_ids=mx.array([target_id], dtype=mx.int32),
            )
        )
    return examples


def build_examples_from_jsonl(tokenizer, path: str, max_examples: int) -> list[TokenExample]:
    examples = []
    with open(path, "r") as fid:
        for line in fid:
            if not line.strip():
                continue
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
            logprobs_vals = row.get("logprobs_vals")
            logprobs_indices = row.get("logprobs_indices")
            examples.append(
                TokenExample(
                    document=document,
                    document_ids=mx.array(tokenizer.encode(document), dtype=mx.int32),
                    prompt_ids=mx.array(prompt_ids, dtype=mx.int32),
                    response_ids=mx.array(response_ids, dtype=mx.int32),
                    logprobs_vals=(
                        mx.array(logprobs_vals, dtype=mx.float32)
                        if logprobs_vals is not None
                        else None
                    ),
                    logprobs_indices=(
                        mx.array(logprobs_indices, dtype=mx.int32)
                        if logprobs_indices is not None
                        else None
                    ),
                )
            )
            if len(examples) >= max_examples:
                break
    if not examples:
        raise ValueError(f"no usable examples found in {path}")
    return examples


def get_vocab_size(tokenizer) -> int:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None and hasattr(tokenizer, "get_vocab"):
        vocab_size = len(tokenizer.get_vocab())
    if vocab_size is None:
        raise ValueError("tokenizer does not expose vocab_size or get_vocab()")
    return int(vocab_size)


def get_token_id(tokenizer, token: str, default: int) -> int:
    token_id = None
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if token_id == unk_token_id:
            token_id = None
    if token_id is None:
        encoded = tokenizer.encode(token)
        if encoded:
            token_id = encoded[-1]
    if token_id is None:
        token_id = default
    return int(token_id)


def module_lookup(model):
    return {
        (spec.layer_idx, spec.module_name): spec
        for spec in infer_lora_module_specs(model, lora_layers=None)
    }


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


def make_loss(model, examples, loss_scope: str, loss_type: str):
    def loss_fn(hypernet):
        losses = []
        for example in examples:
            generated = generate_loras(hypernet, example)
            apply_dynamic_generated_loras(model, generated)
            logits, targets = teacher_forced_response_logits(model, example, loss_scope)
            if loss_type == "kl-topk":
                losses.append(topk_teacher_loss(logits, example, loss_scope))
            else:
                losses.append(nn.losses.cross_entropy(logits, targets).mean())
        return mx.mean(mx.stack(losses))

    return loss_fn


def sample_examples(examples, batch_size: int):
    if batch_size <= 0 or batch_size >= len(examples):
        return examples
    indices = np.random.choice(len(examples), size=batch_size, replace=False)
    return [examples[int(index)] for index in indices]


def topk_teacher_loss(logits: mx.array, example: TokenExample, loss_scope: str) -> mx.array:
    if example.logprobs_vals is None or example.logprobs_indices is None:
        raise ValueError("kl-topk loss requires logprobs_vals and logprobs_indices")
    teacher_vals = example.logprobs_vals
    teacher_indices = example.logprobs_indices
    if loss_scope == "first-token":
        teacher_vals = teacher_vals[:1]
        teacher_indices = teacher_indices[:1]
    token_count = min(logits.shape[0], teacher_vals.shape[0], teacher_indices.shape[0])
    logits = logits[:token_count]
    teacher_vals = teacher_vals[:token_count]
    teacher_indices = teacher_indices[:token_count]
    student_logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    selected_student_logprobs = mx.take_along_axis(
        student_logprobs,
        teacher_indices,
        axis=-1,
    )
    teacher_probs = mx.softmax(teacher_vals, axis=-1)
    return -mx.sum(teacher_probs * selected_student_logprobs, axis=-1).mean()


def teacher_forced_response_logits(model, example, loss_scope: str):
    response_ids = example.response_ids
    if loss_scope == "first-token":
        response_ids = response_ids[:1]
    if len(response_ids) > 1:
        input_ids = mx.concatenate([example.prompt_ids, response_ids[:-1]], axis=0)
    else:
        input_ids = example.prompt_ids
    logits, _ = model(input_ids[None, :])
    start = len(example.prompt_ids) - 1
    end = start + len(response_ids)
    return logits[0, start:end, :].astype(mx.float32), response_ids.astype(mx.int32)


def accuracy(model, hypernet, examples, loss_scope: str):
    correct = 0
    for example in examples:
        generated = generate_loras(hypernet, example)
        apply_dynamic_generated_loras(model, generated)
        logits, targets = teacher_forced_response_logits(model, example, loss_scope)
        preds = mx.argmax(logits, axis=-1)
        correct += int(mx.all(preds == targets).item())
    return correct / len(examples)


def token_accuracy(model, hypernet, examples, loss_scope: str):
    correct = 0
    total = 0
    for example in examples:
        generated = generate_loras(hypernet, example)
        apply_dynamic_generated_loras(model, generated)
        logits, targets = teacher_forced_response_logits(model, example, loss_scope)
        preds = mx.argmax(logits, axis=-1)
        correct += int(mx.sum(preds == targets).item())
        total += len(targets)
    return correct / total if total else 0.0


def response_token_count(examples, loss_scope: str):
    if loss_scope == "first-token":
        return len(examples)
    return sum(len(example.response_ids) for example in examples)


def generate_loras(hypernet, example):
    if example.document_features is not None:
        if isinstance(example.document_features, list):
            generated_groups = [hypernet(features) for features in example.document_features]
            if hasattr(hypernet, "merge_generated_lora_groups"):
                return hypernet.merge_generated_lora_groups(generated_groups)
            return merge_generated_lora_groups(generated_groups)
        return hypernet(example.document_features)
    if isinstance(hypernet, TokenDocToLoRAHypernetwork):
        return hypernet(example.document_ids)
    return hypernet.generate_from_text(example.document)


def chunk_context_ids(document_ids: mx.array, max_context_tokens: int, chunk_tokens: int):
    if max_context_tokens > 0 and len(document_ids) > max_context_tokens:
        document_ids = document_ids[:max_context_tokens]
    if chunk_tokens <= 0:
        return [document_ids]
    return [document_ids[start : start + chunk_tokens] for start in range(0, len(document_ids), chunk_tokens)]


def attach_model_embedding_features(
    model,
    examples,
    max_context_tokens: int,
    chunk_tokens: int,
    keep_sequence: bool = False,
):
    embed = model.model.embed_tokens
    scale = getattr(model.model, "embed_scale", 1.0)
    out = []
    for example in examples:
        features = []
        for context_ids in chunk_context_ids(
            example.document_ids,
            max_context_tokens,
            chunk_tokens,
        ):
            embeddings = embed(context_ids[None, :])[0] * scale
            features.append(embeddings if keep_sequence else mx.mean(embeddings, axis=0))
        document_features = features if len(features) > 1 else features[0]
        out.append(
            TokenExample(
                document=example.document,
                document_ids=example.document_ids,
                prompt_ids=example.prompt_ids,
                response_ids=example.response_ids,
                document_features=document_features,
                logprobs_vals=example.logprobs_vals,
                logprobs_indices=example.logprobs_indices,
            )
        )
    eval_document_features(out)
    return out


def eval_document_features(examples):
    features = []
    for example in examples:
        if isinstance(example.document_features, list):
            features.extend(example.document_features)
        elif example.document_features is not None:
            features.append(example.document_features)
    mx.eval(features)


def attach_model_activation_features(
    model,
    examples,
    max_context_tokens: int,
    chunk_tokens: int,
    activation_pooling: str,
):
    out = []
    for example in examples:
        features = [
            extract_layer_activation_feature(model, context_ids, activation_pooling)
            for context_ids in chunk_context_ids(
                example.document_ids,
                max_context_tokens,
                chunk_tokens,
            )
        ]
        document_features = features if len(features) > 1 else features[0]
        out.append(
            TokenExample(
                document=example.document,
                document_ids=example.document_ids,
                prompt_ids=example.prompt_ids,
                response_ids=example.response_ids,
                document_features=document_features,
                logprobs_vals=example.logprobs_vals,
                logprobs_indices=example.logprobs_indices,
            )
        )
    eval_document_features(out)
    return out


def extract_layer_activation_feature(model, input_ids, activation_pooling: str):
    h = model.model.embed_tokens(input_ids[None, :])
    h = h * getattr(model.model, "embed_scale", 1.0)
    mask = None
    if h.shape[1] > 1:
        mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
        mask = mask.astype(h.dtype)
    layer_features = []
    for layer in model.model.layers:
        h, _ = layer(h, mask, None)
        if activation_pooling == "latent":
            layer_features.append(h[0])
        else:
            layer_features.append(mx.mean(h[0], axis=0))
    return mx.stack(layer_features)


def metrics(model, hypernet, examples, loss_scope: str, loss_type: str):
    if not examples:
        return None
    loss_value = make_loss(model, examples, loss_scope, loss_type)(hypernet).item()
    exact = accuracy(model, hypernet, examples, loss_scope)
    token_acc = token_accuracy(model, hypernet, examples, loss_scope)
    return {
        "loss": loss_value,
        "exact_acc": exact,
        "token_acc": token_acc,
        "response_tokens": response_token_count(examples, loss_scope),
    }


def print_metrics(prefix: str, values: dict | None):
    if values is None:
        return
    print(f"{prefix}_loss={values['loss']:.6f}")
    print(f"{prefix}_exact_acc={values['exact_acc']:.3f}")
    print(f"{prefix}_token_acc={values['token_acc']:.3f}")
    print(f"{prefix}_response_tokens={values['response_tokens']}")


def hypernet_config_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")


def save_hypernet_config(
    checkpoint_path: Path,
    args,
    target_modules: list[str],
    num_specs: int,
) -> None:
    config = {
        "model": args.model,
        "feature_size": args.feature_size,
        "hidden_size": args.hidden_size,
        "rank": args.rank,
        "lora_layers": args.lora_layers,
        "target_modules": ",".join(target_modules),
        "max_specs": args.max_specs,
        "num_specs": num_specs,
        "context_encoder": args.context_encoder,
        "hypernet_aggregator": args.hypernet_aggregator,
        "perceiver_latents": args.perceiver_latents,
        "perceiver_blocks": args.perceiver_blocks,
        "perceiver_self_attn": args.perceiver_self_attn,
        "context_max_tokens": args.context_max_tokens,
        "context_chunk_tokens": args.context_chunk_tokens,
        "chunk_merge": args.chunk_merge,
        "max_context_chunks": args.max_context_chunks,
        "activation_pooling": args.activation_pooling,
        "activation_latents": args.activation_latents,
        "spec_conditioning": args.spec_conditioning,
        "per_rank_gen": args.per_rank_gen,
        "per_layer_processing": args.per_layer_processing,
        "num_pre_head_layers": args.num_pre_head_layers,
        "loss_scope": args.loss_scope,
        "loss_type": args.loss_type,
        "seed": args.seed,
    }
    hypernet_config_path(checkpoint_path).write_text(json.dumps(config, indent=2))


def save_hypernet_checkpoint(
    hypernet,
    checkpoint_path: Path,
    args,
    target_modules: list[str],
    num_specs: int,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(str(checkpoint_path), **dict(tree_flatten(hypernet.parameters())))
    save_hypernet_config(checkpoint_path, args, target_modules, num_specs)


def save_optimizer_checkpoint(optimizer, checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(str(checkpoint_path), **dict(tree_flatten(optimizer.state)))


def load_optimizer_checkpoint(optimizer, checkpoint_path: str) -> None:
    optimizer_state = tree_unflatten(list(mx.load(checkpoint_path).items()))
    optimizer.state.update(optimizer_state)
    mx.eval(optimizer.state)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    model, tokenizer = load_model(args)
    model.freeze()

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    specs = infer_lora_module_specs(
        model,
        target_modules=target_modules,
        lora_layers=args.lora_layers,
    )
    if args.max_specs > 0:
        specs = specs[: args.max_specs]
    if not specs:
        raise SystemExit("no target LoRA modules found")

    hypernet_feature_size = args.feature_size
    if args.context_encoder in {"model-embed", "model-activations"}:
        hypernet_feature_size = model.args.hidden_size

    if args.hypernet_aggregator == "perceiver" and args.context_encoder not in {
        "model-embed",
        "model-activations",
    }:
        raise SystemExit("--hypernet-aggregator perceiver requires model-embed or model-activations")

    if args.context_encoder == "token-hash":
        hypernet = TokenDocToLoRAHypernetwork(
            specs,
            num_buckets=args.context_buckets,
            num_latents=args.context_latents,
            feature_size=hypernet_feature_size,
            hidden_size=args.hidden_size,
            rank=args.rank,
            scale=20.0,
            spec_conditioning=args.spec_conditioning,
            per_rank_gen=args.per_rank_gen,
            per_layer_processing=args.per_layer_processing,
            num_pre_head_layers=args.num_pre_head_layers,
        )
    elif args.hypernet_aggregator == "perceiver":
        hypernet = PerceiverDocToLoRAHypernetwork(
            specs,
            feature_size=hypernet_feature_size,
            hidden_size=args.hidden_size,
            rank=args.rank,
            scale=20.0,
            num_latents=args.perceiver_latents,
            num_blocks=args.perceiver_blocks,
            num_self_attn_per_block=args.perceiver_self_attn,
            per_layer_processing=args.per_layer_processing,
            num_pre_head_layers=args.num_pre_head_layers,
            chunk_merge=args.chunk_merge,
            max_context_chunks=args.max_context_chunks,
        )
    else:
        hypernet = DocToLoRAHypernetwork(
            specs,
            feature_size=hypernet_feature_size,
            hidden_size=args.hidden_size,
            rank=args.rank,
            scale=20.0,
            spec_conditioning=args.spec_conditioning,
            per_rank_gen=args.per_rank_gen,
            per_layer_processing=args.per_layer_processing,
            num_pre_head_layers=args.num_pre_head_layers,
            activation_latents=(
                args.activation_latents
                if args.context_encoder == "model-activations"
                and args.activation_pooling == "latent"
                else 0
            ),
            chunk_merge=args.chunk_merge,
            max_context_chunks=args.max_context_chunks,
        )
    if args.load_hypernet:
        hypernet.load_weights(args.load_hypernet, strict=False)
        mx.eval(hypernet.parameters())
        print(f"loaded_hypernet={args.load_hypernet}")
    examples = build_examples(tokenizer, args)
    if args.context_encoder == "model-embed":
        examples = attach_model_embedding_features(
            model,
            examples,
            args.context_max_tokens,
            args.context_chunk_tokens,
            keep_sequence=args.hypernet_aggregator == "perceiver",
        )
    if args.context_encoder == "model-activations":
        examples = attach_model_activation_features(
            model,
            examples,
            args.context_max_tokens,
            args.context_chunk_tokens,
            args.activation_pooling,
        )
    eval_examples = []
    if args.eval_examples > 0:
        if args.eval_examples >= len(examples):
            raise ValueError("eval_examples must be smaller than total loaded examples")
        eval_examples = examples[-args.eval_examples :]
        examples = examples[: -args.eval_examples]

    optimizer = optim.Adam(learning_rate=args.learning_rate)
    if args.load_optimizer:
        load_optimizer_checkpoint(optimizer, args.load_optimizer)
        print(f"loaded_optimizer={args.load_optimizer}")

    loss_fn = make_loss(model, examples, args.loss_scope, args.loss_type)
    initial_loss = loss_fn(hypernet).item()
    initial_acc = accuracy(model, hypernet, examples, args.loss_scope)
    initial_token_acc = token_accuracy(model, hypernet, examples, args.loss_scope)
    initial_eval = metrics(model, hypernet, eval_examples, args.loss_scope, args.loss_type)
    best_eval_loss = initial_eval["loss"] if initial_eval is not None else math.inf
    saved_best_hypernet = None
    if args.save_best_hypernet and initial_eval is not None:
        best_path = Path(args.save_best_hypernet)
        save_hypernet_checkpoint(hypernet, best_path, args, target_modules, len(specs))
        saved_best_hypernet = best_path
    for step in range(args.iters):
        train_batch = sample_examples(examples, args.batch_size)
        step_loss_fn = make_loss(model, train_batch, args.loss_scope, args.loss_type)
        loss_and_grad = nn.value_and_grad(hypernet, step_loss_fn)
        loss_value, grads = loss_and_grad(hypernet)
        optimizer.update(hypernet, grads)
        mx.eval(hypernet.parameters(), optimizer.state, loss_value)
        if step == 0 or (step + 1) % max(args.iters // 4, 1) == 0:
            print(f"iter {step + 1}: loss={loss_value.item():.6f}")
        if (
            args.save_best_hypernet
            and eval_examples
            and args.eval_every > 0
            and (step + 1) % args.eval_every == 0
        ):
            step_eval = metrics(model, hypernet, eval_examples, args.loss_scope, args.loss_type)
            print(f"iter {step + 1}: eval_loss={step_eval['loss']:.6f}")
            if step_eval["loss"] < best_eval_loss:
                best_eval_loss = step_eval["loss"]
                best_path = Path(args.save_best_hypernet)
                save_hypernet_checkpoint(hypernet, best_path, args, target_modules, len(specs))
                saved_best_hypernet = best_path
                print(f"iter {step + 1}: saved_best_hypernet={best_path}")

    final_loss = loss_fn(hypernet).item()
    if math.isnan(final_loss):
        raise SystemExit("token smoke task produced NaN loss")
    final_acc = accuracy(model, hypernet, examples, args.loss_scope)
    final_token_acc = token_accuracy(model, hypernet, examples, args.loss_scope)
    final_eval = metrics(model, hypernet, eval_examples, args.loss_scope, args.loss_type)
    if args.save_best_hypernet and final_eval is not None and final_eval["loss"] < best_eval_loss:
        best_eval_loss = final_eval["loss"]
        best_path = Path(args.save_best_hypernet)
        save_hypernet_checkpoint(hypernet, best_path, args, target_modules, len(specs))
        saved_best_hypernet = best_path
    improvement = initial_loss / final_loss if final_loss > 0 else math.inf
    print(f"initial_loss={initial_loss:.6f}")
    print(f"final_loss={final_loss:.6f}")
    print(f"improvement={improvement:.2f}x")
    print(f"initial_acc={initial_acc:.3f}")
    print(f"final_acc={final_acc:.3f}")
    print(f"initial_token_acc={initial_token_acc:.3f}")
    print(f"final_token_acc={final_token_acc:.3f}")
    print(f"response_tokens={response_token_count(examples, args.loss_scope)}")
    print_metrics("initial_eval", initial_eval)
    print_metrics("final_eval", final_eval)
    if initial_eval is not None and final_eval is not None:
        eval_improvement = initial_eval["loss"] / final_eval["loss"] if final_eval["loss"] > 0 else math.inf
        print(f"eval_improvement={eval_improvement:.2f}x")
    print(f"target_modules={','.join(target_modules)}")
    print(f"num_specs={len(specs)}")
    print(f"loss_scope={args.loss_scope}")
    print(f"loss_type={args.loss_type}")
    print(f"context_encoder={args.context_encoder}")
    print(f"hypernet_aggregator={args.hypernet_aggregator}")
    print(f"perceiver_latents={args.perceiver_latents}")
    print(f"perceiver_blocks={args.perceiver_blocks}")
    print(f"perceiver_self_attn={args.perceiver_self_attn}")
    print(f"context_latents={args.context_latents}")
    print(f"context_max_tokens={args.context_max_tokens}")
    print(f"context_chunk_tokens={args.context_chunk_tokens}")
    print(f"chunk_merge={args.chunk_merge}")
    print(f"max_context_chunks={args.max_context_chunks}")
    print(f"activation_pooling={args.activation_pooling}")
    print(f"activation_latents={args.activation_latents}")
    print(f"spec_conditioning={args.spec_conditioning}")
    print(f"per_rank_gen={args.per_rank_gen}")
    print(f"per_layer_processing={args.per_layer_processing}")
    print(f"num_pre_head_layers={args.num_pre_head_layers}")
    print(f"batch_size={args.batch_size}")
    print(f"train_examples={len(examples)}")
    print(f"eval_examples={len(eval_examples)}")
    if args.save_hypernet:
        save_path = Path(args.save_hypernet)
        save_hypernet_checkpoint(hypernet, save_path, args, target_modules, len(specs))
        print(f"saved_hypernet={save_path}")
        print(f"saved_hypernet_config={hypernet_config_path(save_path)}")
    if saved_best_hypernet is not None:
        print(f"best_eval_loss={best_eval_loss:.6f}")
        print(f"saved_best_hypernet={saved_best_hypernet}")
        print(f"saved_best_hypernet_config={hypernet_config_path(saved_best_hypernet)}")
    if args.save_optimizer:
        optimizer_path = Path(args.save_optimizer)
        save_optimizer_checkpoint(optimizer, optimizer_path)
        print(f"saved_optimizer={optimizer_path}")

    if args.iters > 0 and final_loss >= initial_loss and final_acc <= initial_acc:
        raise SystemExit("token smoke task did not improve")


if __name__ == "__main__":
    main()
