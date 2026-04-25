#!/usr/bin/env python3
import argparse
import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from lora_mlx.doc_to_lora import DocToLoRAHypernetwork, LoRAModuleSpec


@dataclass(frozen=True)
class SmokeExample:
    document: str
    feature: mx.array
    target_a: mx.array
    target_b: mx.array


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a tiny native MLX Doc-to-LoRA hypernetwork smoke task."
    )
    parser.add_argument("--num-docs", type=int, default=8)
    parser.add_argument("--feature-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--input-dims", type=int, default=12)
    parser.add_argument("--output-dims", type=int, default=10)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save", type=str, default=None)
    return parser.parse_args()


def build_examples(hypernet, args) -> list[SmokeExample]:
    examples = []
    for doc_id in range(args.num_docs):
        document = (
            f"Synthetic document {doc_id}. "
            f"The private memory key is KEY-{doc_id:02d}. "
            f"The adapter should encode answer slot {doc_id}."
        )
        rng = np.random.default_rng(args.seed + doc_id)
        target_a = mx.array(
            rng.normal(0, 0.25, size=(args.input_dims, args.rank)).astype(np.float32)
        )
        target_b = mx.array(
            rng.normal(0, 0.25, size=(args.rank, args.output_dims)).astype(np.float32)
        )
        examples.append(
            SmokeExample(
                document=document,
                feature=hypernet.encode_text(document),
                target_a=target_a,
                target_b=target_b,
            )
        )
    return examples


def make_probe_inputs(args):
    rng = np.random.default_rng(args.seed + 10_000)
    shape = (args.num_docs, 16, args.input_dims)
    return mx.array(rng.normal(0, 1, size=shape).astype(np.float32))


def make_loss(examples, probe_inputs, scale: float):
    def loss_fn(hypernet):
        losses = []
        for doc_idx, example in enumerate(examples):
            generated = hypernet(example.feature)[0]
            x = probe_inputs[doc_idx]
            predicted = (x @ generated.lora_a) @ generated.lora_b * scale
            target = (x @ example.target_a) @ example.target_b * scale
            losses.append(mx.mean(mx.square(predicted - target)))
        return mx.mean(mx.stack(losses))

    return loss_fn


def generated_delta_norm(hypernet, example, probe_x):
    generated = hypernet(example.feature)[0]
    delta = (probe_x @ generated.lora_a) @ generated.lora_b * generated.scale
    return mx.sqrt(mx.mean(mx.square(delta))).item()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    spec = LoRAModuleSpec(
        layer_idx=0,
        module_name="q_proj",
        input_dims=args.input_dims,
        output_dims=args.output_dims,
    )
    hypernet = DocToLoRAHypernetwork(
        [spec],
        feature_size=args.feature_size,
        hidden_size=args.hidden_size,
        rank=args.rank,
        scale=1.0,
    )
    examples = build_examples(hypernet, args)
    probe_inputs = make_probe_inputs(args)
    loss_fn = make_loss(examples, probe_inputs, scale=1.0)
    loss_and_grad = nn.value_and_grad(hypernet, loss_fn)
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    initial_loss = loss_fn(hypernet).item()
    for step in range(args.iters):
        loss_value, grads = loss_and_grad(hypernet)
        optimizer.update(hypernet, grads)
        mx.eval(hypernet.parameters(), optimizer.state, loss_value)
        if step == 0 or (step + 1) % max(args.iters // 4, 1) == 0:
            print(f"iter {step + 1}: loss={loss_value.item():.6f}")

    final_loss = loss_fn(hypernet).item()
    improvement = initial_loss / final_loss if final_loss > 0 else math.inf
    delta_norm = generated_delta_norm(hypernet, examples[0], probe_inputs[0])
    print(f"initial_loss={initial_loss:.6f}")
    print(f"final_loss={final_loss:.6f}")
    print(f"improvement={improvement:.2f}x")
    print(f"generated_delta_norm={delta_norm:.6f}")

    if args.save:
        mx.savez(args.save, **dict(tree_flatten(hypernet.parameters())))
        print(f"saved={args.save}")

    if final_loss >= initial_loss * 0.35:
        raise SystemExit("smoke task did not learn enough; final loss is too high")


if __name__ == "__main__":
    main()
