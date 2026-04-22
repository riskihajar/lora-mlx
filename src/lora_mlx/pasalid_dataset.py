import argparse
import json
import random
import re
from pathlib import Path

from .pasalid import DEFAULT_PASAL_ID_RAW_DIR
from .paths import DEFAULT_DATA_DIR


DEFAULT_PASAL_ID_DATASET_DIR = DEFAULT_DATA_DIR / "pasalid"


def load_law_detail_files(raw_dir: Path) -> list[dict]:
    payloads = []
    for path in sorted(raw_dir.glob("*.json")):
        if path.name == "laws_index.json":
            continue
        payloads.append(json.loads(path.read_text()))
    return payloads


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def clean_content(text: str) -> str:
    text = text.replace("\u20ac", "")
    text = re.sub(r"REPUBLIK\s+INDONESIA", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"REPUBLIK\s*\]NDONES\]A", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"LTIFFIT\.FN\s+INDONESIA", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bBAB\s+[IVXLCDM]+\b.*$", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bPasal\s+[A-Za-z0-9.-]+\s*$", " ", text)
    text = re.sub(r"\.{2,}", " ", text)
    text = re.sub(r"\s+-\d+\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_content_usable(content: str) -> bool:
    if len(content) < 80:
        return False
    lowered = content.lower()
    noisy_markers = [
        "republik indonesia",
        "line truncated",
        "... (line truncated",
        " . .",
    ]
    if any(marker in lowered for marker in noisy_markers):
        return False

    alpha_chars = sum(char.isalpha() for char in content)
    digit_chars = sum(char.isdigit() for char in content)
    if alpha_chars < 40:
        return False
    if digit_chars > alpha_chars:
        return False

    weird_spacing = len(re.findall(r"\b\w\s{1,2}\w\s{1,2}\w\b", content))
    if weird_spacing > 3:
        return False

    return True


def article_reference(work: dict, article: dict) -> str:
    title = work.get("title", "")
    prefix = f"Undang-Undang Nomor {work.get('number', '')} Tahun {work.get('year', '')} tentang "
    if title.startswith(prefix):
        title = title[len(prefix) :]
    return (
        f"{work.get('type', '')} No. {work.get('number', '')} Tahun {work.get('year', '')} "
        f"tentang {title}, Pasal {article.get('number', '')}"
    ).strip()


def build_question_variants(work: dict, article: dict) -> list[str]:
    title = work.get("title", "")
    article_number = article.get("number", "")
    content = normalize_space(article.get("content", ""))
    first_sentence = content.split(".", 1)[0].strip() if content else ""

    prompts = [
        f"Apa isi {article_reference(work, article)}?",
        f"Jelaskan bunyi Pasal {article_number} dalam {title}.",
    ]

    if first_sentence:
        prompts.append(
            f"Dalam {title}, pasal mana yang memuat ketentuan berikut: {first_sentence}?"
        )

    return prompts


def build_answer(work: dict, article: dict) -> str:
    reference = article_reference(work, article)
    content = normalize_space(article.get("content", ""))
    return f"{reference}: {content}"


def build_examples_from_payload(payload: dict) -> list[dict]:
    work = payload.get("work", {})
    if not work.get("content_verified", False):
        return []
    title = work.get("title", "")
    if re.fullmatch(r"Undang-Undang Nomor \d+ Tahun \d+ tentang Undang-Undang Nomor \d+ Tahun \d+", title):
        return []
    examples = []
    for article in payload.get("articles", []):
        if article.get("type") != "pasal":
            continue
        content = clean_content(normalize_space(article.get("content", "") or ""))
        if not is_content_usable(content):
            continue

        article = dict(article)
        article["content"] = content

        answer = build_answer(work, article)
        for prompt in build_question_variants(work, article):
            examples.append(
                {
                    "text": f"Pertanyaan: {prompt}\nJawaban: {answer}",
                    "metadata": {
                        "frbr_uri": work.get("frbr_uri"),
                        "title": work.get("title"),
                        "article_number": article.get("number"),
                    },
                }
            )
    return examples


def split_examples_by_law(examples: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict], list[str], list[str], list[str]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = {}
    for example in examples:
        frbr_uri = example["metadata"]["frbr_uri"]
        grouped.setdefault(frbr_uri, []).append(example)

    law_ids = list(grouped.keys())
    rng.shuffle(law_ids)

    total_laws = len(law_ids)
    train_end = max(1, int(total_laws * 0.8))
    valid_end = max(train_end + 1, int(total_laws * 0.9)) if total_laws >= 3 else total_laws

    train_laws = law_ids[:train_end]
    valid_laws = law_ids[train_end:valid_end]
    test_laws = law_ids[valid_end:]

    if not valid_laws and len(train_laws) > 1:
        valid_laws = [train_laws.pop()]
    if not test_laws:
        if len(valid_laws) > 1:
            test_laws = [valid_laws.pop()]
        elif len(train_laws) > 1:
            test_laws = [train_laws.pop()]

    train_rows = [row for law_id in train_laws for row in grouped[law_id]]
    valid_rows = [row for law_id in valid_laws for row in grouped[law_id]]
    test_rows = [row for law_id in test_laws for row in grouped[law_id]]

    return train_rows, valid_rows, test_rows, train_laws, valid_laws, test_laws


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps({"text": row["text"]}, ensure_ascii=True) + "\n")


def build_dataset(raw_dir: Path, output_dir: Path, seed: int) -> dict[str, int]:
    payloads = load_law_detail_files(raw_dir)
    examples = []
    for payload in payloads:
        examples.extend(build_examples_from_payload(payload))

    train_rows, valid_rows, test_rows, train_laws, valid_laws, test_laws = split_examples_by_law(examples, seed)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "valid.jsonl", valid_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)

    manifest = {
        "total_examples": len(examples),
        "train_examples": len(train_rows),
        "valid_examples": len(valid_rows),
        "test_examples": len(test_rows),
        "total_laws": len(set(example["metadata"]["frbr_uri"] for example in examples)),
        "train_laws": train_laws,
        "valid_laws": valid_laws,
        "test_laws": test_laws,
        "source_raw_dir": str(raw_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a pilot Pasal.id QA dataset from local raw cache.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_PASAL_ID_RAW_DIR), help="Pasal.id raw cache directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_PASAL_ID_DATASET_DIR), help="Output dataset directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_dataset(Path(args.raw_dir), Path(args.output_dir), args.seed)
    print(f"total_examples={manifest['total_examples']}")
    print(f"train_examples={manifest['train_examples']}")
    print(f"valid_examples={manifest['valid_examples']}")
    print(f"test_examples={manifest['test_examples']}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
