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
from mlx.utils import tree_flatten

from lora_mlx import utils as lora_utils
from lora_mlx.doc_to_lora import infer_lora_module_specs
from lora_mlx.models import LoRALinear


@dataclass(frozen=True)
class BaselineExample:
    document_ids: mx.array
    prompt_ids: mx.array
    response_ids: mx.array


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate an ordinary LoRA baseline on Sakana JSONL splits."
    )
    parser.add_argument("--model", default="mlx-community/gemma-2-2b-it")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of examples per optimizer step. Keep small to avoid large MLX graphs.",
    )
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--lora-layers", type=int, default=2)
    parser.add_argument("--target-modules", default="down_proj")
    parser.add_argument("--max-specs", type=int, default=2)
    parser.add_argument("--load-adapter", default=None)
    parser.add_argument("--save-adapter", default=None)
    parser.add_argument("--save-best-adapter", default=None)
    parser.add_argument("--loss-scope", choices=["first-token", "full-answer"], default="full-answer")
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def load_examples(tokenizer, path: str, max_examples: int) -> list[BaselineExample]:
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
                BaselineExample(
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


def apply_lora_to_target_modules(model, args):
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
    for spec in specs:
        layer = model.model.layers[spec.layer_idx]
        if spec.module_name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            current = getattr(layer.self_attn, spec.module_name)
            lora = LoRALinear.from_linear(current, rank=args.rank)
            lora.scale = args.scale
            setattr(layer.self_attn, spec.module_name, lora)
        elif spec.module_name in {"gate_proj", "down_proj", "up_proj"}:
            current = getattr(layer.mlp, spec.module_name)
            lora = LoRALinear.from_linear(current, rank=args.rank)
            lora.scale = args.scale
            setattr(layer.mlp, spec.module_name, lora)
        else:
            raise ValueError(f"unsupported module {spec.module_name!r}")
    return specs


def teacher_forced_logits(model, example: BaselineExample, loss_scope: str, source_context: bool = False):
    response_ids = example.response_ids
    if loss_scope == "first-token":
        response_ids = response_ids[:1]
    prompt_ids = example.prompt_ids
    if source_context:
        prompt_ids = mx.concatenate([example.document_ids, prompt_ids], axis=0)
    if len(response_ids) > 1:
        input_ids = mx.concatenate([prompt_ids, response_ids[:-1]], axis=0)
    else:
        input_ids = prompt_ids
    logits, _ = model(input_ids[None, :])
    start = len(prompt_ids) - 1
    end = start + len(response_ids)
    return logits[0, start:end, :].astype(mx.float32), response_ids.astype(mx.int32)


def make_loss(examples: list[BaselineExample], loss_scope: str):
    def loss_fn(model):
        losses = []
        for example in examples:
            logits, targets = teacher_forced_logits(model, example, loss_scope)
            losses.append(nn.losses.cross_entropy(logits, targets).mean())
        return mx.mean(mx.stack(losses))

    return loss_fn


def sample_batch(examples: list[BaselineExample], batch_size: int):
    if batch_size >= len(examples):
        return examples
    indices = np.random.choice(len(examples), size=batch_size, replace=False)
    return [examples[int(index)] for index in indices]


def metrics(model, examples: list[BaselineExample], loss_scope: str, source_context: bool = False):
    if not examples:
        return None
    losses = []
    correct = 0
    total = 0
    exact = 0
    for example in examples:
        logits, targets = teacher_forced_logits(model, example, loss_scope, source_context)
        loss = nn.losses.cross_entropy(logits, targets).mean()
        losses.append(loss.item())
        preds = mx.argmax(logits, axis=-1)
        correct += int(mx.sum(preds == targets).item())
        total += len(targets)
        exact += int(mx.all(preds == targets).item())
    return {
        "loss": float(np.mean(losses)),
        "token_acc": correct / total if total else 0.0,
        "exact_acc": exact / len(examples),
        "response_tokens": total,
    }


def print_metrics(prefix: str, values: dict | None):
    if values is None:
        return
    print(f"{prefix}_loss={values['loss']:.6f}")
    print(f"{prefix}_exact_acc={values['exact_acc']:.3f}")
    print(f"{prefix}_token_acc={values['token_acc']:.3f}")
    print(f"{prefix}_response_tokens={values['response_tokens']}")


def save_adapter(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(str(path), **dict(tree_flatten(model.trainable_parameters())))


def main():
    args = parse_args()
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    model, tokenizer, _ = lora_utils.load(args.model)
    model.freeze()
    specs = apply_lora_to_target_modules(model, args)
    mx.eval(model.parameters())
    if args.load_adapter:
        model.load_weights(args.load_adapter, strict=False)
        mx.eval(model.parameters())
        print(f"loaded_adapter={args.load_adapter}")

    examples = load_examples(tokenizer, args.dataset_jsonl, args.max_examples)
    eval_examples = []
    if args.eval_examples > 0:
        if args.eval_examples >= len(examples):
            raise ValueError("eval_examples must be smaller than total loaded examples")
        eval_examples = examples[-args.eval_examples :]
        examples = examples[: -args.eval_examples]

    optimizer = optim.Adam(learning_rate=args.learning_rate)

    initial = metrics(model, examples, args.loss_scope)
    initial_eval = metrics(model, eval_examples, args.loss_scope)
    source_eval = metrics(model, eval_examples, args.loss_scope, source_context=True)
    best_eval_loss = initial_eval["loss"] if initial_eval is not None else math.inf
    saved_best_adapter = None
    if args.save_best_adapter and initial_eval is not None:
        best_path = Path(args.save_best_adapter)
        save_adapter(model, best_path)
        saved_best_adapter = best_path

    for step in range(args.iters):
        loss_fn = make_loss(sample_batch(examples, args.batch_size), args.loss_scope)
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss_value, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), optimizer.state, loss_value)
        if step == 0 or (step + 1) % max(args.iters // 4, 1) == 0:
            print(f"iter {step + 1}: loss={loss_value.item():.6f}")
        if args.save_best_adapter and eval_examples and args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            step_eval = metrics(model, eval_examples, args.loss_scope)
            print(f"iter {step + 1}: eval_loss={step_eval['loss']:.6f}")
            if step_eval["loss"] < best_eval_loss:
                best_eval_loss = step_eval["loss"]
                best_path = Path(args.save_best_adapter)
                save_adapter(model, best_path)
                saved_best_adapter = best_path
                print(f"iter {step + 1}: saved_best_adapter={best_path}")

    final = metrics(model, examples, args.loss_scope)
    final_eval = metrics(model, eval_examples, args.loss_scope)
    if args.save_best_adapter and final_eval is not None and final_eval["loss"] < best_eval_loss:
        best_eval_loss = final_eval["loss"]
        best_path = Path(args.save_best_adapter)
        save_adapter(model, best_path)
        saved_best_adapter = best_path

    print_metrics("initial", initial)
    print_metrics("final", final)
    print_metrics("initial_eval", initial_eval)
    print_metrics("source_context_eval", source_eval)
    print_metrics("final_eval", final_eval)
    if initial is not None and final is not None:
        improvement = initial["loss"] / final["loss"] if final["loss"] > 0 else math.inf
        print(f"improvement={improvement:.2f}x")
    if initial_eval is not None and final_eval is not None:
        eval_improvement = initial_eval["loss"] / final_eval["loss"] if final_eval["loss"] > 0 else math.inf
        print(f"eval_improvement={eval_improvement:.2f}x")
    print(f"target_modules={args.target_modules}")
    print(f"num_specs={len(specs)}")
    print(f"rank={args.rank}")
    print(f"loss_scope={args.loss_scope}")
    print(f"batch_size={args.batch_size}")
    print(f"train_examples={len(examples)}")
    print(f"eval_examples={len(eval_examples)}")

    if args.save_adapter:
        adapter_path = Path(args.save_adapter)
        save_adapter(model, adapter_path)
        print(f"saved_adapter={adapter_path}")
    if saved_best_adapter is not None:
        print(f"best_eval_loss={best_eval_loss:.6f}")
        print(f"saved_best_adapter={saved_best_adapter}")

    if args.iters > 0 and final["loss"] >= initial["loss"]:
        raise SystemExit("ordinary LoRA baseline did not improve")


if __name__ == "__main__":
    main()
