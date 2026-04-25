#!/usr/bin/env python3

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from lora_mlx.paths import DEFAULT_DATA_DIR


DEFAULT_DOC_UNITS = DEFAULT_DATA_DIR / "pasalid" / "doc_units.jsonl"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "pasalid" / "qa_bank_natural_legal.jsonl"
DEFAULT_SPLIT_DIR = DEFAULT_DATA_DIR / "pasalid" / "natural_legal_split"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def max_contiguous_copy_run(answer: str, source_doc: str) -> int:
    answer_tokens = tokenize(answer)
    source_tokens = tokenize(source_doc)
    if not answer_tokens or not source_tokens:
        return 0
    source_positions = defaultdict(list)
    for index, token in enumerate(source_tokens):
        source_positions[token].append(index)

    best = 0
    for answer_index, token in enumerate(answer_tokens):
        for source_index in source_positions.get(token, []):
            run = 0
            while (
                answer_index + run < len(answer_tokens)
                and source_index + run < len(source_tokens)
                and answer_tokens[answer_index + run] == source_tokens[source_index + run]
            ):
                run += 1
            best = max(best, run)
    return best


def is_usable(row: dict, max_source_chars: int) -> bool:
    source_doc = normalize_space(str(row.get("source_doc", "")))
    article = str(row.get("article_number", "")).strip()
    if len(source_doc) < 50 or not article:
        return False
    if max_source_chars > 0 and len(source_doc) > max_source_chars:
        return False
    lowered = source_doc.lower()
    noisy_markers = [
        "line truncated",
        " . .",
        "merr,",
        "darrr",
        "i[",
        "ihdonesta",
        "repu].ik",
        "repueuk",
        "repuellik",
        "repuelik",
    ]
    if any(marker in lowered for marker in noisy_markers):
        return False

    financial_markers = [
        "laporan keuangan",
        "laporan realisasi anggaran",
        "laporan perubahan",
        "laporan operasional",
        "neraca",
        "arus kas",
    ]
    if any(marker in lowered for marker in financial_markers):
        return False
    budget_markers = [
        "apbn",
        "saldo anggaran",
        "pendapatan negara",
        "belanja negara",
        "pembiayaan anggaran",
        "postur apbn",
    ]
    if "tahun anggaran" in lowered and any(marker in lowered for marker in budget_markers):
        return False
    if "apbn terdiri atas anggaran" in lowered:
        return False
    if "tahun anggaran" in lowered and re.search(r"rp\s*[0-9so][0-9so\.\s,|]{6,}", lowered):
        return False
    if "aporan operasional" in lowered or "aporan perubahan" in lowered:
        return False

    return True


def source_payload(row: dict, answer: str) -> str:
    payload = {
        "answer": normalize_space(answer),
        "source_type": str(row.get("type", "UU")),
        "source_number": str(row.get("number", "")),
        "source_year": str(row.get("year", "")),
        "source_article": str(row.get("article_number", "")).strip(),
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def build_row(row: dict, question: str, answer: str, generation: str) -> dict:
    return {
        "law_id": row["law_id"],
        "frbr_uri": row.get("frbr_uri", row["law_id"]),
        "title": row.get("title", ""),
        "short_title": row.get("short_title", ""),
        "article_number": str(row.get("article_number", "")).strip(),
        "source_reference": row.get("source_reference", ""),
        "source_doc": normalize_space(row.get("source_doc", "")),
        "question": normalize_space(question),
        "answer": source_payload(row, answer),
        "answer_style": "natural_paraphrase_with_structured_citation",
        "source_generation": generation,
        "max_source_copy_run": max_contiguous_copy_run(answer, row.get("source_doc", "")),
    }


def title_subject(row: dict) -> str:
    return normalize_space(row.get("short_title") or row.get("title") or "ketentuan ini")


def fallback_examples(row: dict) -> list[tuple[str, str]]:
    source_doc = normalize_space(row.get("source_doc", ""))
    lowered = source_doc.lower()
    subject = title_subject(row)
    reference = row.get("source_reference", "sumber hukum terkait")

    if "ancaman kekerasan" in lowered and "dipidana" in lowered:
        return [
            (
                "Kalau seseorang mengirim ancaman lewat media elektronik, apa konsekuensi hukumnya?",
                f"Perbuatan mengirim ancaman melalui media elektronik dapat berujung pidana. Berdasarkan {reference}, ancamannya adalah pidana penjara dan/atau denda sesuai batas yang disebut dalam ketentuan tersebut.",
            )
        ]
    if "nama baik" in lowered or "kehormatan" in lowered:
        return [
            (
                "Apa aturan jika seseorang mencemarkan nama baik orang lain lewat sistem elektronik?",
                f"Tindakan menyerang kehormatan atau nama baik melalui informasi elektronik termasuk perbuatan yang diatur secara khusus. Dasar rujukannya adalah {reference}.",
            )
        ]
    if "kecamatan" in lowered and "terdiri" in lowered:
        return [
            (
                f"Apa saja kecamatan yang termasuk dalam wilayah {subject}?",
                f"Wilayah tersebut dibagi ke dalam sejumlah kecamatan sebagaimana dirinci dalam {reference}. Rinciannya harus mengikuti daftar kecamatan dalam ketentuan tersebut.",
            )
        ]
    if "yang dimaksud dengan" in lowered or "adalah" in lowered:
        return [
            (
                f"Apa definisi penting yang dipakai dalam {subject}?",
                f"Ketentuan definisi dalam {reference} menjelaskan istilah yang dipakai dalam aturan tersebut agar ruang lingkup pengaturannya jelas.",
            )
        ]
    return [
        (
            f"Apa isu hukum utama yang diatur dalam {subject}?",
            f"Ketentuan ini mengatur isu hukum yang dijelaskan dalam {reference}. Jawaban substantif harus dibaca bersama dokumen sumber agar tidak keluar dari ruang lingkup pasal tersebut.",
        )
    ]


def targeted_transition_examples(row: dict) -> list[tuple[str, str]]:
    source_doc = normalize_space(row.get("source_doc", ""))
    lowered = source_doc.lower()
    reference = row.get("source_reference", "sumber hukum terkait")
    subject = title_subject(row)
    examples = []

    if "masih tetap berlaku" in lowered and "sepanjang tidak bertentangan" in lowered:
        examples.extend(
            [
                (
                    f"Apakah aturan pelaksanaan lama terkait {subject} masih berlaku setelah aturan baru ini ada?",
                    f"Masih berlaku, tetapi hanya sepanjang aturan pelaksanaan lama itu tidak bertentangan dengan ketentuan dalam aturan baru. Rujukannya adalah {reference}.",
                ),
                (
                    "Kalau ada aturan turunan lama, apakah otomatis gugur atau tetap bisa dipakai?",
                    f"Aturan turunan lama tidak otomatis gugur. Aturan itu tetap bisa dipakai selama tidak bertentangan dengan ketentuan baru sebagaimana disebut dalam {reference}.",
                ),
            ]
        )

    if "dicabut" in lowered and "dinyatakan tidak berlaku" in lowered:
        examples.extend(
            [
                (
                    f"Apakah ketentuan lama tentang {subject} masih berlaku setelah aturan baru ini berlaku?",
                    f"Tidak. Ketentuan lama yang mengatur hal tersebut dicabut dan dinyatakan tidak berlaku berdasarkan {reference}.",
                ),
                (
                    "Apa nasib aturan lama yang disebut dalam pasal peralihan ini?",
                    f"Aturan lama yang disebut dalam pasal tersebut tidak lagi berlaku karena sudah dicabut dan dinyatakan tidak berlaku. Dasarnya adalah {reference}.",
                ),
            ]
        )

    return examples


def llm_examples(row: dict, questions_per_doc: int) -> list[tuple[str, str]]:
    from lora_mlx.openai_compat import generate_text

    system_prompt = (
        "Anda membangun dataset tesis untuk retrieval-grounded legal QA Indonesia. "
        "Buat pertanyaan seperti user nyata, bukan pertanyaan template tentang 'dokumen ini'. "
        "Jawaban harus berupa parafrase natural 1-3 kalimat, grounded hanya pada dokumen sumber, "
        "tidak menyalin mentah pasal, dan tidak memberi nasihat hukum di luar sumber. "
        "Kembalikan JSON array valid berisi object dengan field question dan answer."
    )
    user_prompt = json.dumps(
        {
            "source_reference": row.get("source_reference", ""),
            "title": row.get("title", ""),
            "source_doc": normalize_space(row.get("source_doc", "")),
            "n": questions_per_doc,
            "rules": [
                "question harus natural dan bisa ditanyakan tanpa melihat dokumen sumber",
                "answer harus menyebut substansi hukum dengan bahasa sendiri",
                "answer tidak boleh menyalin lebih dari 10 kata berurutan dari source_doc",
                "jangan buat pertanyaan yang jawabannya tidak ada di source_doc",
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
    text = generate_text(system_prompt, user_prompt)
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    try:
        values = json.loads(match.group(0) if match else text)
    except json.JSONDecodeError:
        return []
    examples = []
    for value in values:
        if not isinstance(value, dict):
            continue
        question = normalize_space(str(value.get("question", "")))
        answer = normalize_space(str(value.get("answer", "")))
        if question and answer:
            examples.append((question, answer))
    return examples[:questions_per_doc]


def build_rows(docs: list[dict], limit: int, questions_per_doc: int, use_llm: bool, max_copy_run: int) -> list[dict]:
    rows = []
    seen = set()
    generation = "llm_natural_paraphrase" if use_llm else "heuristic_natural_bootstrap"
    for doc in docs:
        targeted = targeted_transition_examples(doc)
        remaining = max(0, questions_per_doc - len(targeted))
        generated = llm_examples(doc, remaining) if use_llm and remaining else fallback_examples(doc)
        examples = (targeted + generated)[:questions_per_doc]
        for question, answer in examples:
            key = (question, doc.get("source_reference", ""))
            if key in seen:
                continue
            if max_contiguous_copy_run(answer, doc.get("source_doc", "")) > max_copy_run:
                continue
            seen.add(key)
            rows.append(build_row(doc, question, answer, generation))
            if len(rows) >= limit:
                return rows
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_split(path: Path, rows: list[dict], include_source_doc: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            text = f"Q: {row['question']}\nA: {row['answer']}"
            if include_source_doc:
                text = (
                    f"Dokumen sumber: {row['source_doc']}\n"
                    f"Referensi: {row['source_reference']}\n"
                    "Instruksi: Jawab dengan bahasa Indonesia natural berdasarkan dokumen sumber. "
                    "Jangan menyalin mentah dokumen. Tetap sertakan sumber dalam JSON.\n"
                    f"Q: {row['question']}\n"
                    f"A: {row['answer']}"
                )
            handle.write(json.dumps({"text": text}, ensure_ascii=True) + "\n")


def split_rows(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    by_law = defaultdict(list)
    for row in rows:
        by_law[row["law_id"]].append(row)
    law_ids = list(by_law)
    random.Random(seed).shuffle(law_ids)
    if len(law_ids) < 4:
        raise ValueError("Need at least 4 laws for natural legal QA split.")
    valid_law = law_ids[0]
    unseen_law = next((law_id for law_id in law_ids[1:] if len(by_law[law_id]) >= 20), law_ids[1])
    valid_laws = {valid_law}
    unseen_laws = {unseen_law}
    train_laws = [law_id for law_id in law_ids if law_id not in valid_laws | unseen_laws]

    train = []
    test_seen = []
    rng = random.Random(seed)
    for law_id in train_laws:
        law_rows = by_law[law_id][:]
        rng.shuffle(law_rows)
        split_at = max(1, int(len(law_rows) * 0.75))
        if split_at >= len(law_rows):
            split_at = max(0, len(law_rows) - 1)
        train.extend(law_rows[:split_at])
        test_seen.extend(law_rows[split_at:])
    return {
        "train": train,
        "valid": [row for law_id in valid_laws for row in by_law[law_id]],
        "test_seen": test_seen,
        "test_unseen": [row for law_id in unseen_laws for row in by_law[law_id]],
    }


def write_outputs(rows: list[dict], output: Path, split_dir: Path, seed: int) -> None:
    write_jsonl(output, rows)
    splits = split_rows(rows, seed)
    write_split(split_dir / "train.jsonl", splits["train"], include_source_doc=True)
    write_split(split_dir / "valid.jsonl", splits["valid"], include_source_doc=True)
    write_split(split_dir / "test_seen.jsonl", splits["test_seen"], include_source_doc=False)
    write_split(split_dir / "test_seen_with_context.jsonl", splits["test_seen"], include_source_doc=True)
    write_split(split_dir / "test_unseen.jsonl", splits["test_unseen"], include_source_doc=False)
    write_split(split_dir / "test_unseen_with_context.jsonl", splits["test_unseen"], include_source_doc=True)
    manifest = {
        "rows": len(rows),
        "laws": len({row["law_id"] for row in rows}),
        "answer_style": "natural_paraphrase_with_structured_citation",
        "avg_max_source_copy_run": sum(row["max_source_copy_run"] for row in rows) / len(rows),
        "splits": {name: len(split_rows) for name, split_rows in splits.items()},
    }
    (split_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build natural paraphrased Pasal.id legal QA for core LoRA experiments.")
    parser.add_argument("--input", default=str(DEFAULT_DOC_UNITS), help="Input doc_units JSONL")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output QA bank JSONL")
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR), help="Output split directory")
    parser.add_argument("--limit", type=int, default=120, help="Maximum generated QA rows")
    parser.add_argument("--questions-per-doc", type=int, default=2, help="Questions to request per doc when using LLM")
    parser.add_argument("--max-source-chars", type=int, default=3000, help="Maximum source_doc character length")
    parser.add_argument("--max-copy-run", type=int, default=10, help="Reject answers copying more than N contiguous source tokens")
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument("--use-llm", action="store_true", help="Use configured OpenAI-compatible model for natural QA generation")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    docs = [row for row in load_rows(Path(args.input)) if is_usable(row, args.max_source_chars)]
    rows = build_rows(docs, args.limit, args.questions_per_doc, args.use_llm, args.max_copy_run)
    if not rows:
        raise ValueError("No natural QA rows generated. Try --use-llm or relax filters.")
    write_outputs(rows, Path(args.output), Path(args.split_dir), args.seed)


if __name__ == "__main__":
    main()
