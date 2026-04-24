#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from lora_mlx.paths import DEFAULT_ADAPTERS_DIR, DEFAULT_DATA_DIR


DEFAULT_DOC_UNITS = DEFAULT_DATA_DIR / "pasalid" / "doc_units.jsonl"
DEFAULT_OUTPUT = DEFAULT_ADAPTERS_DIR / "adapters_pasalid_hyperproto_tinyllama_mixture.npz"
DEFAULT_BASIS = [
    DEFAULT_ADAPTERS_DIR / "adapters_pasalid_tinyllama_experiment.npz",
    DEFAULT_ADAPTERS_DIR / "adapters_pasalid_tinyllama_final.npz",
    DEFAULT_ADAPTERS_DIR / "adapters_pasalid_tinyllama_final_400.npz",
]


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


def validate_basis(basis_weights: list[dict]) -> list[str]:
    keys = list(basis_weights[0].keys())
    key_set = set(keys)
    for index, weights in enumerate(basis_weights[1:], start=1):
        if set(weights.keys()) != key_set:
            raise ValueError(f"Basis adapter {index} has different tensor keys")
        for key in keys:
            if tuple(weights[key].shape) != tuple(basis_weights[0][key].shape):
                raise ValueError(f"Basis adapter {index} tensor {key} has incompatible shape")
    return keys


def coefficients_from_text(text: str, count: int, temperature: float) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = []
    counter = 0
    while len(raw) < count:
        block = hashlib.sha256(digest + counter.to_bytes(4, "little")).digest()
        raw.extend(int.from_bytes(block[i : i + 4], "little") / 2**32 for i in range(0, len(block), 4))
        counter += 1
    logits = np.array(raw[:count], dtype=np.float64)
    logits = (logits - logits.mean()) / max(temperature, 1e-6)
    logits = logits - logits.max()
    probs = np.exp(logits)
    return probs / probs.sum()


def parse_coefficients(value: str, count: int) -> np.ndarray:
    coefficients = np.array([float(part.strip()) for part in value.split(",") if part.strip()], dtype=np.float64)
    if len(coefficients) != count:
        raise ValueError(f"Expected {count} coefficients, got {len(coefficients)}")
    total = coefficients.sum()
    if not math.isfinite(total) or total == 0:
        raise ValueError("Coefficient sum must be finite and non-zero")
    return coefficients / total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a document-conditioned mixture of existing LoRA basis adapters.")
    parser.add_argument("--basis", nargs="+", default=[str(path) for path in DEFAULT_BASIS], help="Basis adapter .npz files with identical tensor schema")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output mixed adapter .npz")
    parser.add_argument("--manifest", default=None, help="Optional manifest path; defaults to output path with .json suffix")
    parser.add_argument("--doc-units", default=str(DEFAULT_DOC_UNITS), help="Doc units JSONL used when --doc-text/--doc-file are omitted")
    parser.add_argument("--doc-index", type=int, default=0, help="Doc unit index used as prototype context")
    parser.add_argument("--doc-file", default=None, help="Optional text file used as prototype context")
    parser.add_argument("--doc-text", default=None, help="Optional literal text used as prototype context")
    parser.add_argument("--coefficients", default=None, help="Optional comma-separated coefficients. If omitted, deterministic doc-conditioned coefficients are used")
    parser.add_argument("--temperature", type=float, default=0.5, help="Softmax temperature for doc-conditioned coefficients")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    basis_paths = [Path(path) for path in args.basis]
    basis_weights = [mx.load(str(path)) for path in basis_paths]
    keys = validate_basis(basis_weights)

    doc_text = load_doc_text(args)
    coefficients = (
        parse_coefficients(args.coefficients, len(basis_paths))
        if args.coefficients
        else coefficients_from_text(doc_text, len(basis_paths), args.temperature)
    )

    mixed = {}
    for key in keys:
        tensor = sum(float(weight) * basis[key] for weight, basis in zip(coefficients, basis_weights))
        mixed[key] = tensor.astype(mx.float32)

    output_path = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(str(output_path), **mixed)

    manifest = {
        "adapter_type": "doc_to_lora_mixture_prototype",
        "basis": [str(path) for path in basis_paths],
        "coefficients": [float(value) for value in coefficients],
        "output": str(output_path),
        "doc_sha256": hashlib.sha256(doc_text.encode("utf-8")).hexdigest(),
        "doc_chars": len(doc_text),
        "tensor_count": len(mixed),
        "parameter_count": int(sum(value.size for value in mixed.values())),
        "note": "Document-conditioned mixture prototype; coefficients are deterministic unless provided explicitly.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
