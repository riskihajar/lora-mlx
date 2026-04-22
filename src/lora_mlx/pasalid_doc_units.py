import argparse
import json
import re
from pathlib import Path

from .pasalid import DEFAULT_PASAL_ID_RAW_DIR
from .paths import DEFAULT_DATA_DIR


DEFAULT_PASAL_ID_DOC_UNITS = DEFAULT_DATA_DIR / "pasalid" / "doc_units.jsonl"


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
    noisy_markers = ["line truncated", "... (line truncated", " . ."]
    if any(marker in lowered for marker in noisy_markers):
        return False
    alpha_chars = sum(char.isalpha() for char in content)
    digit_chars = sum(char.isdigit() for char in content)
    if alpha_chars < 40:
        return False
    if digit_chars > alpha_chars:
        return False
    return True


def strip_title_prefix(title: str, number: str, year: int) -> str:
    prefix = f"Undang-Undang Nomor {number} Tahun {year} tentang "
    if title.startswith(prefix):
        return title[len(prefix) :]
    return title


def infer_question_type(content: str) -> str:
    lowered = content.lower()
    if "yang dimaksud dengan" in lowered:
        return "definition"
    if "wajib" in lowered:
        return "obligation"
    if "dilarang" in lowered:
        return "prohibition"
    if "dipidana" in lowered or "pidana" in lowered or "denda" in lowered:
        return "sanction"
    if "terdiri atas" in lowered or "merupakan" in lowered:
        return "scope"
    return "article_qa"


def build_doc_units(raw_dir: Path) -> list[dict]:
    units = []
    for path in sorted(raw_dir.glob("*.json")):
        if path.name == "laws_index.json":
            continue
        payload = json.loads(path.read_text())
        work = payload.get("work", {})
        if not work.get("content_verified", False):
            continue
        title = work.get("title", "")
        if re.fullmatch(
            r"Undang-Undang Nomor \d+ Tahun \d+ tentang Undang-Undang Nomor \d+ Tahun \d+",
            title,
        ):
            continue

        short_title = strip_title_prefix(title, str(work.get("number", "")), work.get("year", ""))

        for article in payload.get("articles", []):
            if article.get("type") != "pasal":
                continue
            content = clean_content(normalize_space(article.get("content", "") or ""))
            if not is_content_usable(content):
                continue

            units.append(
                {
                    "law_id": work.get("frbr_uri"),
                    "frbr_uri": work.get("frbr_uri"),
                    "title": title,
                    "short_title": short_title,
                    "type": work.get("type"),
                    "number": work.get("number"),
                    "year": work.get("year"),
                    "article_number": article.get("number"),
                    "question_type_candidate": infer_question_type(content),
                    "source_reference": f"{work.get('type')} No. {work.get('number')} Tahun {work.get('year')}, Pasal {article.get('number')}",
                    "source_doc": content,
                }
            )
    return units


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build canonical Pasal.id document units from raw cache.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_PASAL_ID_RAW_DIR), help="Pasal.id raw cache directory")
    parser.add_argument("--output", default=str(DEFAULT_PASAL_ID_DOC_UNITS), help="Output JSONL file for canonical doc units")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    units = build_doc_units(Path(args.raw_dir))
    write_jsonl(Path(args.output), units)
    print(f"doc_units={len(units)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
