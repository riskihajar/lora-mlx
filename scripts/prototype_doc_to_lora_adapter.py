#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from lora_mlx.paths import DEFAULT_ADAPTERS_DIR, DEFAULT_DATA_DIR


DEFAULT_TEMPLATE = DEFAULT_ADAPTERS_DIR / "adapters_pasalid_tinyllama_experiment.npz"
DEFAULT_DOC_UNITS = DEFAULT_DATA_DIR / "pasalid" / "doc_units.jsonl"
DEFAULT_OUTPUT = DEFAULT_ADAPTERS_DIR / "adapters_pasalid_hyperproto_tinyllama.npz"


def load_doc_text(args: argparse.Namespace) -> str:
    if args.doc_text:
        return args.doc_text
    if args.doc_file:
        return Path(args.doc_file).read_text()

    rows = [json.loads(line) for line in Path(args.doc_units).read_text().splitlines() if line.strip()]
    if args.doc_index < 0 or args.doc_index >= len(rows):
        raise ValueError(f"doc_index must be within [0, {len(rows) - 1}]")
    row = rows[args.doc_index]
    return "\n".join(
        [
            str(row.get("title", "")),
            str(row.get("source_reference", "")),
            str(row.get("source_doc", "")),
        ]
    ).strip()


def seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def generated_tensor(key: str, shape: tuple[int, ...], rng: np.random.Generator, mode: str, output_scale: float) -> mx.array:
    if mode == "zero":
        return mx.zeros(shape, dtype=mx.float32)

    if key.endswith("lora_a") and len(shape) == 2:
        input_dims = max(shape[0], 1)
        std = 1 / math.sqrt(input_dims)
    elif key.endswith("lora_b"):
        std = output_scale
    else:
        std = output_scale

    values = rng.normal(loc=0.0, scale=std, size=shape).astype(np.float32)
    return mx.array(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a Doc-to-LoRA-style prototype adapter with the existing MLX LoRA schema.")
    parser.add_argument("--template-adapter", default=str(DEFAULT_TEMPLATE), help="Existing adapter used only for tensor keys and shapes")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output generated adapter .npz")
    parser.add_argument("--manifest", default=None, help="Optional manifest path; defaults to output path with .json suffix")
    parser.add_argument("--doc-units", default=str(DEFAULT_DOC_UNITS), help="Doc units JSONL used when --doc-text/--doc-file are omitted")
    parser.add_argument("--doc-index", type=int, default=0, help="Doc unit index used as prototype context")
    parser.add_argument("--doc-file", default=None, help="Optional text file used as prototype context")
    parser.add_argument("--doc-text", default=None, help="Optional literal text used as prototype context")
    parser.add_argument("--mode", choices=["zero", "hash"], default="zero", help="Generation mode. zero validates loading; hash creates deterministic tiny LoRA weights")
    parser.add_argument("--output-scale", type=float, default=1e-5, help="Stddev for generated non-A LoRA weights in hash mode")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    template_path = Path(args.template_adapter)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(".json")

    doc_text = load_doc_text(args)
    seed = seed_from_text(doc_text)
    rng = np.random.default_rng(seed)
    template = mx.load(str(template_path))
    generated = {
        key: generated_tensor(key, tuple(value.shape), rng, args.mode, args.output_scale)
        for key, value in template.items()
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(str(output_path), **generated)

    manifest = {
        "adapter_type": "doc_to_lora_prototype",
        "mode": args.mode,
        "template_adapter": str(template_path),
        "output": str(output_path),
        "doc_sha256": hashlib.sha256(doc_text.encode("utf-8")).hexdigest(),
        "doc_chars": len(doc_text),
        "seed": seed,
        "tensor_count": len(generated),
        "parameter_count": int(sum(value.size for value in generated.values())),
        "note": "Integration prototype only; not a trained hypernetwork checkpoint.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
