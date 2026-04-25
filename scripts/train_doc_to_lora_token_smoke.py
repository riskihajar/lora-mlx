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

from lora_mlx import utils as lora_utils
from lora_mlx.doc_to_lora import (
    DocToLoRAHypernetwork,
    GeneratedLoRALinear,
    TokenDocToLoRAHypernetwork,
    infer_lora_module_specs,
)
from lora_mlx.models import Model, ModelArgs


@dataclass(frozen=True)
class TokenExample:
    document: str
    document_ids: mx.array
    prompt_ids: mx.array
    response_ids: mx.array


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
        choices=["hash", "token-hash"],
        default="hash",
        help="Use deterministic text hash features or trainable hashed token features.",
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lora-layers", type=int, default=1)
    parser.add_argument("--target-modules", default="down_proj")
    parser.add_argument("--iters", type=int, default=60)
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
        "--target-token-prefix",
        default="the",
        help="Tokenizer token used as the first synthetic target; later docs offset from it.",
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
            examples.append(
                TokenExample(
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


def make_loss(model, examples, loss_scope: str):
    def loss_fn(hypernet):
        losses = []
        for example in examples:
            generated = generate_loras(hypernet, example)
            apply_dynamic_generated_loras(model, generated)
            logits, targets = teacher_forced_response_logits(model, example, loss_scope)
            losses.append(nn.losses.cross_entropy(logits, targets).mean())
        return mx.mean(mx.stack(losses))

    return loss_fn


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
    if isinstance(hypernet, TokenDocToLoRAHypernetwork):
        return hypernet(example.document_ids)
    return hypernet.generate_from_text(example.document)


def metrics(model, hypernet, examples, loss_scope: str):
    if not examples:
        return None
    loss_value = make_loss(model, examples, loss_scope)(hypernet).item()
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

    if args.context_encoder == "token-hash":
        hypernet = TokenDocToLoRAHypernetwork(
            specs,
            num_buckets=args.context_buckets,
            num_latents=args.context_latents,
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            rank=args.rank,
            scale=20.0,
            spec_conditioning=args.spec_conditioning,
            per_rank_gen=args.per_rank_gen,
            per_layer_processing=args.per_layer_processing,
            num_pre_head_layers=args.num_pre_head_layers,
        )
    else:
        hypernet = DocToLoRAHypernetwork(
            specs,
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            rank=args.rank,
            scale=20.0,
            spec_conditioning=args.spec_conditioning,
            per_rank_gen=args.per_rank_gen,
            per_layer_processing=args.per_layer_processing,
            num_pre_head_layers=args.num_pre_head_layers,
        )
    examples = build_examples(tokenizer, args)
    eval_examples = []
    if args.eval_examples > 0:
        if args.eval_examples >= len(examples):
            raise ValueError("eval_examples must be smaller than total loaded examples")
        eval_examples = examples[-args.eval_examples :]
        examples = examples[: -args.eval_examples]

    loss_fn = make_loss(model, examples, args.loss_scope)
    loss_and_grad = nn.value_and_grad(hypernet, loss_fn)
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    initial_loss = loss_fn(hypernet).item()
    initial_acc = accuracy(model, hypernet, examples, args.loss_scope)
    initial_token_acc = token_accuracy(model, hypernet, examples, args.loss_scope)
    initial_eval = metrics(model, hypernet, eval_examples, args.loss_scope)
    for step in range(args.iters):
        loss_value, grads = loss_and_grad(hypernet)
        optimizer.update(hypernet, grads)
        mx.eval(hypernet.parameters(), optimizer.state, loss_value)
        if step == 0 or (step + 1) % max(args.iters // 4, 1) == 0:
            print(f"iter {step + 1}: loss={loss_value.item():.6f}")

    final_loss = loss_fn(hypernet).item()
    if math.isnan(final_loss):
        raise SystemExit("token smoke task produced NaN loss")
    final_acc = accuracy(model, hypernet, examples, args.loss_scope)
    final_token_acc = token_accuracy(model, hypernet, examples, args.loss_scope)
    final_eval = metrics(model, hypernet, eval_examples, args.loss_scope)
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
    print(f"context_encoder={args.context_encoder}")
    print(f"context_latents={args.context_latents}")
    print(f"spec_conditioning={args.spec_conditioning}")
    print(f"per_rank_gen={args.per_rank_gen}")
    print(f"per_layer_processing={args.per_layer_processing}")
    print(f"num_pre_head_layers={args.num_pre_head_layers}")
    print(f"train_examples={len(examples)}")
    print(f"eval_examples={len(eval_examples)}")

    if final_loss >= initial_loss and final_acc <= initial_acc:
        raise SystemExit("token smoke task did not improve")


if __name__ == "__main__":
    main()
