import inspect
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelArgs:
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    rope_theta: float = 10000
    rope_traditional: bool = False
    model_type: Optional[str] = None
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    hidden_activation: str = "silu"
    sliding_window: Optional[int] = None
    layer_types: Optional[list[str]] = None
    global_head_dim: Optional[int] = None
    rope_parameters: Optional[Dict[str, Dict[str, Union[float, str]]]] = None
    hidden_size_per_layer_input: Optional[int] = None
    vocab_size_per_layer_input: Optional[int] = None
    final_logit_softcapping: Optional[float] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] != "linear":
                raise ValueError("rope_scaling 'type' currently only supports 'linear'")

        if self.global_head_dim is None:
            self.global_head_dim = self.head_dim

        if self.vocab_size_per_layer_input is None:
            self.vocab_size_per_layer_input = self.vocab_size

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def _gelu_pytorch_tanh(x):
    return nn.gelu_approx(x)


def _activation(name: str):
    if name in {"silu", "swish"}:
        return nn.silu
    if name == "gelu_pytorch_tanh":
        return _gelu_pytorch_tanh
    if name == "gelu":
        return nn.gelu
    raise ValueError(f"Unsupported activation {name}")


def _dequantize_embedding_weight(embedding):
    weight = embedding.weight
    if isinstance(embedding, nn.QuantizedEmbedding):
        weight = mx.dequantize(
            weight,
            embedding.scales,
            embedding.biases,
            embedding.group_size,
            embedding.bits,
        )
    return weight


class RMSNormNoScale(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, None, self.eps)


class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(linear: nn.Linear, rank: int = 8):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = LoRALinear(input_dims, output_dims, rank)
        lora_lin.linear = linear
        return lora_lin

    def to_linear(self):
        linear = self.linear
        bias = "bias" in linear
        weight = linear.weight
        is_quantized = isinstance(linear, nn.QuantizedLinear)

        dtype = weight.dtype

        if is_quantized:
            dtype = mx.float16
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.biases,
                linear.group_size,
                linear.bits,
            )
        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)

        lora_b = (self.scale * self.lora_b.T).astype(dtype)
        lora_a = self.lora_a.T.astype(dtype)
        fused_linear.weight = weight + lora_b @ lora_a
        if bias:
            fused_linear.bias = linear.bias

        if is_quantized:
            fused_linear = nn.QuantizedLinear.from_linear(
                fused_linear,
                linear.group_size,
                linear.bits,
            )

        return fused_linear

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        lora_rank: int = 8,
        bias: bool = False,
        scale: float = 20.0,
    ):
        super().__init__()

        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.scale = scale

        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, lora_rank),
        )
        self.lora_b = mx.zeros(shape=(lora_rank, output_dims))

    def __call__(self, x):
        dtype = self.linear.weight.dtype
        if isinstance(self.linear, nn.QuantizedLinear):
            dtype = self.linear.scales.dtype
        y = self.linear(x.astype(dtype))
        z = (x @ self.lora_a) @ self.lora_b
        return y + self.scale * z


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.repeats = n_heads // n_kv_heads

        head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.q_norm = None
        self.k_norm = None
        if args.model_type == "qwen3":
            self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        rope_scale = (
            1 / float(args.rope_scaling["factor"])
            if args.rope_scaling is not None and args.rope_scaling["type"] == "linear"
            else 1
        )
        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
            scale=rope_scale,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if self.q_norm is not None:
            queries = self.q_norm(queries)
        if self.k_norm is not None:
            keys = self.k_norm(keys)

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, activation: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.act = _activation(activation)

    def __call__(self, x) -> mx.array:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size, args.hidden_activation)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache


class GemmaAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = None if args.layer_types is None else args.layer_types[layer_idx]
        self.is_full_attention = self.layer_type == "full_attention"
        self.head_dim = args.global_head_dim if self.is_full_attention else args.head_dim
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.repeats = self.n_heads // self.n_kv_heads
        self.scale = 1.0
        self.sliding_window = args.sliding_window if self.layer_type == "sliding_attention" else None

        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.v_norm = RMSNormNoScale(self.head_dim, eps=args.rms_norm_eps)

        rope_params = None
        if args.rope_parameters is not None and self.layer_type is not None:
            rope_params = args.rope_parameters.get(self.layer_type)
        rope_base = args.rope_theta if rope_params is None else rope_params.get("rope_theta", args.rope_theta)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)

    def _apply_sliding_mask(self, scores, q_len: int, k_len: int):
        if self.sliding_window is None:
            return scores
        q_positions = mx.arange(q_len)[:, None]
        k_positions = mx.arange(k_len)[None, :]
        lower = k_positions <= q_positions
        upper = (q_positions - k_positions) < self.sliding_window
        allowed = lower & upper
        mask = mx.where(allowed, 0.0, -1e9).astype(scores.dtype)
        return scores + mask[None, None, :, :]

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)
        values = self.v_norm(values)

        if cache is not None:
            key_cache, value_cache = cache
            offset = key_cache.shape[2]
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if self.repeats > 1:
            keys = mx.repeat(keys, self.repeats, axis=1)
            values = mx.repeat(values, self.repeats, axis=1)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask.astype(scores.dtype)
        scores = self._apply_sliding_mask(scores, queries.shape[2], keys.shape[2])
        probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(values.dtype)
        output = probs @ values
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys[:, : self.n_kv_heads], values[:, : self.n_kv_heads])


class GemmaTransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = GemmaAttention(args, layer_idx)
        self.mlp = MLP(args.hidden_size, args.intermediate_size, args.hidden_activation)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_feedforward_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layer_scalar = mx.ones((1,))
        self.args = args

        self.hidden_size_per_layer_input = args.hidden_size_per_layer_input or 0
        if self.hidden_size_per_layer_input:
            self.act = _activation(args.hidden_activation)
            self.per_layer_input_gate = nn.Linear(
                args.hidden_size, self.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = nn.Linear(
                self.hidden_size_per_layer_input, args.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )

    def __call__(self, x, per_layer_input=None, mask=None, cache=None):
        residual = x
        attn_out, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        x = residual + self.post_attention_layernorm(attn_out)

        residual = x
        mlp_out = self.mlp(self.pre_feedforward_layernorm(x))
        x = residual + self.post_feedforward_layernorm(mlp_out)

        if self.hidden_size_per_layer_input and per_layer_input is not None:
            residual = x
            ple = self.per_layer_input_gate(x)
            ple = self.act(ple)
            ple = ple * per_layer_input
            ple = self.per_layer_projection(ple)
            ple = self.post_per_layer_input_norm(ple)
            x = residual + ple

        x = x * self.layer_scalar
        return x, cache


class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args=args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.norm(h), cache


class GemmaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_scale = math.sqrt(args.hidden_size)
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.embed_tokens_per_layer = None
        self.per_layer_input_scale = 2.0**-0.5
        self.per_layer_model_projection_scale = args.hidden_size**-0.5
        if args.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = nn.Embedding(
                args.vocab_size_per_layer_input,
                args.num_hidden_layers * args.hidden_size_per_layer_input,
            )
            self.per_layer_model_projection = nn.Linear(
                args.hidden_size,
                args.num_hidden_layers * args.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_projection_norm = nn.RMSNorm(
                args.hidden_size_per_layer_input, eps=args.rms_norm_eps
            )
        self.layers = [GemmaTransformerBlock(args=args, layer_idx=i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def _get_per_layer_inputs(self, inputs, h):
        if self.embed_tokens_per_layer is None:
            return None
        token_identity = self.embed_tokens_per_layer(inputs)
        token_identity = token_identity.reshape(
            inputs.shape[0],
            inputs.shape[1],
            self.args.num_hidden_layers,
            self.args.hidden_size_per_layer_input,
        )
        token_identity = token_identity * math.sqrt(self.args.hidden_size_per_layer_input)
        projection = self.per_layer_model_projection(h) * self.per_layer_model_projection_scale
        projection = projection.reshape(
            h.shape[0],
            h.shape[1],
            self.args.num_hidden_layers,
            self.args.hidden_size_per_layer_input,
        )
        projection = self.per_layer_projection_norm(projection)
        return (projection + token_identity) * self.per_layer_input_scale

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs) * self.embed_scale

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1]).astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        per_layer_inputs = self._get_per_layer_inputs(inputs, h)
        for e, layer in enumerate(self.layers):
            layer_input = None if per_layer_inputs is None else per_layer_inputs[:, :, e, :]
            h, cache[e] = layer(h, layer_input, mask, cache[e])

        return self.norm(h), cache


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        if args.model_type == "gemma4":
            self.model = GemmaModel(args)
        else:
            self.model = LlamaModel(args)
        self.lm_head = None
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out, cache = self.model(inputs, cache)
        if self.lm_head is None:
            weight = _dequantize_embedding_weight(self.model.embed_tokens)
            logits = out @ weight.T
        else:
            logits = self.lm_head(out)
        if self.args.final_logit_softcapping is not None:
            logits = logits / self.args.final_logit_softcapping
            logits = mx.tanh(logits)
            logits = logits * self.args.final_logit_softcapping
        return logits, cache
