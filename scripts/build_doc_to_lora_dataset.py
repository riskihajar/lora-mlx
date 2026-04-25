#!/usr/bin/env python3

import argparse
import json
import random
import re
from pathlib import Path


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def slugify(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-._")
    return value.lower() or "document"


def read_document(path: Path) -> str:
    text = path.read_text(errors="ignore")
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    return normalize_space(text)


def split_sentences(text: str) -> list[str]:
    candidates = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = []
    for candidate in candidates:
        sentence = normalize_space(candidate)
        if 40 <= len(sentence) <= 700:
            sentences.append(sentence)
    return sentences


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end]
        if end < len(text):
            last_period = max(chunk.rfind("."), chunk.rfind(";"), chunk.rfind("\n"))
            if last_period > max_chars // 2:
                end = start + last_period + 1
                chunk = text[start:end]
        chunks.append(normalize_space(chunk))
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)
    return [chunk for chunk in chunks if chunk]


def extract_subject(sentence: str, fallback_title: str) -> str:
    match = re.search(r"^([A-Z0-9][^,.]{3,90}?)\s+(?:adalah|merupakan|ialah|is|are|was|were)\b", sentence)
    if match:
        return normalize_space(match.group(1))
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9._/-]*", fallback_title)
    return " ".join(words[:8]) or "dokumen ini"


def sentence_examples(sentence: str, title: str, index: int) -> list[tuple[str, str, str]]:
    subject = extract_subject(sentence, title)
    examples = [
        (
            f"Apa fakta penting yang disebut dokumen tentang {subject}?",
            sentence,
            "sentence_fact",
        )
    ]
    if re.search(r"\b(adalah|merupakan|ialah|is|are|was|were)\b", sentence, flags=re.IGNORECASE):
        examples.append((f"Apa yang dijelaskan tentang {subject}?", sentence, "definition_like_fact"))
    if re.search(r"\b\d{2,4}\b|\b\d+[.,]?\d*\b", sentence):
        examples.append((f"Angka atau tanggal apa yang perlu diingat dari dokumen bagian {index}?", sentence, "numeric_fact"))
    return examples


def chunk_examples(chunk: str, title: str, index: int) -> list[tuple[str, str, str]]:
    return [
        (
            f"Ringkas fakta utama dari dokumen '{title}' bagian {index}.",
            chunk,
            "chunk_recall",
        )
    ]


def build_examples(document: str, title: str, max_chars: int, overlap_chars: int, max_examples: int) -> list[dict]:
    examples = []
    chunks = chunk_text(document, max_chars=max_chars, overlap_chars=overlap_chars)
    for index, chunk in enumerate(chunks, start=1):
        for question, answer, kind in chunk_examples(chunk, title, index):
            examples.append({"question": question, "answer": answer, "kind": kind, "chunk_index": index})
        for sentence in split_sentences(chunk):
            for question, answer, kind in sentence_examples(sentence, title, index):
                examples.append({"question": question, "answer": answer, "kind": kind, "chunk_index": index})
    deduped = []
    seen = set()
    for example in examples:
        key = (example["question"].lower(), example["answer"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)
    return deduped[:max_examples]


def format_example(example: dict, title: str) -> str:
    return (
        "Anda adalah model yang sudah menginternalisasi dokumen berikut sebagai memori LoRA. "
        "Jawab berdasarkan memori dokumen, tanpa meminta konteks tambahan.\n"
        f"Dokumen: {title}\n"
        f"Q: {example['question']}\n"
        f"A: {example['answer']}"
    )


def split_examples(examples: list[dict], seed: int) -> dict[str, list[dict]]:
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 3:
        raise ValueError("Need at least 3 generated examples for train/valid/test split.")
    train_end = max(1, int(len(shuffled) * 0.8))
    valid_end = max(train_end + 1, int(len(shuffled) * 0.9))
    if valid_end >= len(shuffled):
        valid_end = len(shuffled) - 1
    return {
        "train": shuffled[:train_end],
        "valid": shuffled[train_end:valid_end],
        "test": shuffled[valid_end:],
    }


def write_split(output_dir: Path, name: str, rows: list[dict], title: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{name}.jsonl").open("w") as handle:
        for row in rows:
            payload = {"text": format_example(row, title), **row}
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a working document-to-LoRA supervised memory dataset.")
    parser.add_argument("--input", required=True, help="Input .txt/.md document")
    parser.add_argument("--output-dir", default="", help="Output split directory; defaults to data/doc_to_lora/<document-stem>")
    parser.add_argument("--title", default="", help="Human-readable document title")
    parser.add_argument("--max-chars", type=int, default=1400, help="Max chars per recall chunk")
    parser.add_argument("--overlap-chars", type=int, default=180, help="Overlapping chars between chunks")
    parser.add_argument("--max-examples", type=int, default=400, help="Maximum generated QA examples")
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    title = args.title or input_path.stem.replace("_", " ").replace("-", " ").strip() or "document"
    output_dir = Path(args.output_dir) if args.output_dir else Path("data/doc_to_lora") / slugify(input_path.stem)
    document = read_document(input_path)
    examples = build_examples(document, title, args.max_chars, args.overlap_chars, args.max_examples)
    splits = split_examples(examples, args.seed)
    for name, rows in splits.items():
        write_split(output_dir, name, rows, title)
    manifest = {
        "input": str(input_path),
        "title": title,
        "output_dir": str(output_dir),
        "examples": len(examples),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "kind_counts": {kind: sum(1 for row in examples if row["kind"] == kind) for kind in sorted({row["kind"] for row in examples})},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
