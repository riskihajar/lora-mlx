import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .models import LoRALinear


DEFAULT_TARGET_MODULES = ("q_proj", "v_proj")


@dataclass(frozen=True)
class LoRAModuleSpec:
    layer_idx: int
    module_name: str
    input_dims: int
    output_dims: int


@dataclass(frozen=True)
class GeneratedLoRA:
    layer_idx: int
    module_name: str
    lora_a: mx.array
    lora_b: mx.array
    scale: float


class HashContextEncoder:
    """Small deterministic text encoder for D2L plumbing smoke tests."""

    def __init__(self, feature_size: int = 256):
        if feature_size <= 0:
            raise ValueError("feature_size must be positive")
        self.feature_size = feature_size

    def __call__(self, text: str) -> mx.array:
        features = np.zeros((self.feature_size,), dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.feature_size
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            features[bucket] += sign
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return mx.array(features)


class DocToLoRAHypernetwork(nn.Module):
    """Generate LoRA matrices from a context feature vector.

    This is the first native MLX skeleton for SakanaAI-style D2L. It validates the
    `context -> generated A/B -> patched forward` path, but it is not yet the full
    Perceiver/context-activation architecture used by SakanaAI.
    """

    def __init__(
        self,
        module_specs: Sequence[LoRAModuleSpec],
        feature_size: int = 256,
        hidden_size: int = 512,
        rank: int = 8,
        scale: float = 20.0,
    ):
        super().__init__()
        if not module_specs:
            raise ValueError("module_specs must not be empty")
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.module_specs = list(module_specs)
        self.rank = rank
        self.scale = scale
        self.encoder = HashContextEncoder(feature_size)
        self.proj = nn.Linear(feature_size, hidden_size)
        self.a_heads = [
            nn.Linear(hidden_size, spec.input_dims * rank) for spec in self.module_specs
        ]
        self.b_heads = [
            nn.Linear(hidden_size, rank * spec.output_dims) for spec in self.module_specs
        ]

    def encode_text(self, text: str) -> mx.array:
        return self.encoder(text)

    def __call__(self, context_features: mx.array) -> list[GeneratedLoRA]:
        if len(context_features.shape) == 1:
            context_features = context_features[None, :]
        if context_features.shape[0] != 1:
            raise ValueError("only batch size 1 is supported by the initial D2L skeleton")

        hidden = nn.silu(self.proj(context_features))[0]
        generated = []
        init_scale = 1 / math.sqrt(hidden.shape[-1])
        for spec, a_head, b_head in zip(self.module_specs, self.a_heads, self.b_heads):
            lora_a = a_head(hidden).reshape(spec.input_dims, self.rank) * init_scale
            lora_b = b_head(hidden).reshape(self.rank, spec.output_dims) * init_scale
            generated.append(
                GeneratedLoRA(
                    layer_idx=spec.layer_idx,
                    module_name=spec.module_name,
                    lora_a=lora_a,
                    lora_b=lora_b,
                    scale=self.scale,
                )
            )
        return generated

    def generate_from_text(self, text: str) -> list[GeneratedLoRA]:
        return self(self.encode_text(text))


class HashedTokenContextEncoder(nn.Module):
    """Trainable mean-pooled token encoder with bounded embedding size."""

    def __init__(self, num_buckets: int = 4096, feature_size: int = 256):
        super().__init__()
        if num_buckets <= 0:
            raise ValueError("num_buckets must be positive")
        self.num_buckets = num_buckets
        self.embedding = nn.Embedding(num_buckets, feature_size)
        self.norm = nn.RMSNorm(feature_size)

    def __call__(self, token_ids: mx.array) -> mx.array:
        if len(token_ids.shape) != 1:
            token_ids = token_ids.reshape(-1)
        token_ids = token_ids.astype(mx.int32) % self.num_buckets
        x = self.embedding(token_ids)
        return self.norm(mx.mean(x, axis=0))


class TokenDocToLoRAHypernetwork(nn.Module):
    """Generate LoRA matrices from trainable token-level context features."""

    def __init__(
        self,
        module_specs: Sequence[LoRAModuleSpec],
        num_buckets: int = 4096,
        feature_size: int = 256,
        hidden_size: int = 512,
        rank: int = 8,
        scale: float = 20.0,
    ):
        super().__init__()
        self.context_encoder = HashedTokenContextEncoder(num_buckets, feature_size)
        self.hypernet = DocToLoRAHypernetwork(
            module_specs,
            feature_size=feature_size,
            hidden_size=hidden_size,
            rank=rank,
            scale=scale,
        )

    def __call__(self, context_ids: mx.array) -> list[GeneratedLoRA]:
        return self.hypernet(self.context_encoder(context_ids))


class GeneratedLoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, generated_lora: GeneratedLoRA):
        super().__init__()
        self.linear = linear
        self.lora_a = generated_lora.lora_a
        self.lora_b = generated_lora.lora_b
        self.scale = generated_lora.scale

    def __call__(self, x):
        dtype = self.linear.weight.dtype
        if isinstance(self.linear, nn.QuantizedLinear):
            dtype = self.linear.scales.dtype
        y = self.linear(x.astype(dtype))
        z = (x @ self.lora_a.astype(x.dtype)) @ self.lora_b.astype(x.dtype)
        return y + self.scale * z


def infer_lora_module_specs(
    model,
    target_modules: Iterable[str] = DEFAULT_TARGET_MODULES,
    lora_layers: int | None = None,
) -> list[LoRAModuleSpec]:
    layers = model.model.layers
    start_idx = 0 if lora_layers is None else max(len(layers) - lora_layers, 0)
    target_modules = tuple(target_modules)
    specs = []
    for layer_idx, layer in enumerate(layers[start_idx:], start=start_idx):
        for module_name in target_modules:
            module = _get_transformer_module(layer, module_name)
            if module is None:
                continue
            linear = _unwrap_linear(module)
            output_dims, input_dims = linear.weight.shape
            if isinstance(linear, nn.QuantizedLinear):
                input_dims *= 32 // linear.bits
            specs.append(
                LoRAModuleSpec(
                    layer_idx=layer_idx,
                    module_name=module_name,
                    input_dims=input_dims,
                    output_dims=output_dims,
                )
            )
    return specs


def apply_generated_loras(model, generated_loras: Sequence[GeneratedLoRA]) -> None:
    layers = model.model.layers
    for generated_lora in generated_loras:
        layer = layers[generated_lora.layer_idx]
        module = _get_transformer_module(layer, generated_lora.module_name)
        if module is None:
            raise ValueError(
                f"module {generated_lora.module_name!r} not found in layer {generated_lora.layer_idx}"
            )
        linear = _unwrap_linear(module)
        patched = GeneratedLoRALinear(linear, generated_lora)
        _set_transformer_module(layer, generated_lora.module_name, patched)


def _unwrap_linear(module):
    if isinstance(module, GeneratedLoRALinear):
        return module.linear
    if isinstance(module, LoRALinear):
        return module.linear
    return module


def _get_transformer_module(layer, module_name: str):
    if module_name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return getattr(layer.self_attn, module_name, None)
    if module_name in {"gate_proj", "down_proj", "up_proj"}:
        return getattr(layer.mlp, module_name, None)
    raise ValueError(f"unsupported target module {module_name!r}")


def _set_transformer_module(layer, module_name: str, module) -> None:
    if module_name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        setattr(layer.self_attn, module_name, module)
        return
    if module_name in {"gate_proj", "down_proj", "up_proj"}:
        setattr(layer.mlp, module_name, module)
        return
    raise ValueError(f"unsupported target module {module_name!r}")
