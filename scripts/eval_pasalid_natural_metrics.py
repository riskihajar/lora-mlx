#!/usr/bin/env python3

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from lora_mlx.eval_legal import (
    citation_component_score,
    citation_exact_match,
    split_answer_and_source,
    token_f1,
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def max_contiguous_copy_run(answer: str, source_doc: str) -> int:
    answer_tokens = tokenize(answer)
    source_tokens = tokenize(source_doc)
    if not answer_tokens or not source_tokens:
        return 0
    source_positions = {}
    for index, token in enumerate(source_tokens):
        source_positions.setdefault(token, []).append(index)
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


def ngram_set(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def source_ngram_copy_ratio(answer: str, source_doc: str, n: int = 4) -> float:
    answer_ngrams = ngram_set(tokenize(answer), n)
    if not answer_ngrams:
        return 0.0
    source_ngrams = ngram_set(tokenize(source_doc), n)
    return len(answer_ngrams & source_ngrams) / len(answer_ngrams)


def valid_json_rate(prediction: str) -> int:
    text = prediction.strip()
    if not text.startswith("{") or not text.endswith("}"):
        return 0
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return 0
    return int(isinstance(value, dict) and "answer" in value)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(path: Path) -> dict[str, float]:
    rows = load_rows(path)
    answer_f1s = []
    citation_ems = []
    citation_components = []
    copy_runs = []
    copy_ratios = []
    json_rates = []

    for row in rows:
        prediction = row.get("prediction", row.get("pred", ""))
        gold = row["gold"]
        source_doc = row.get("source_doc", "")
        if not source_doc and row.get("prompt", "").startswith("Dokumen sumber: "):
            source_doc = row["prompt"].split("Dokumen sumber: ", 1)[1].split("\nReferensi: ", 1)[0]

        pred_answer, pred_source = split_answer_and_source(prediction)
        gold_answer, gold_source = split_answer_and_source(gold)
        answer_f1s.append(token_f1(pred_answer, gold_answer))
        citation_ems.append(citation_exact_match(pred_source, gold_source))
        citation_components.append(citation_component_score(pred_source, gold_source))
        copy_runs.append(max_contiguous_copy_run(pred_answer, source_doc))
        copy_ratios.append(source_ngram_copy_ratio(pred_answer, source_doc))
        json_rates.append(valid_json_rate(prediction))

    return {
        "examples": len(rows),
        "answer_f1": mean(answer_f1s),
        "citation_em": mean(citation_ems),
        "citation_component_score": mean(citation_components),
        "valid_json_rate": mean(json_rates),
        "avg_max_source_copy_run": mean(copy_runs),
        "avg_source_4gram_copy_ratio": mean(copy_ratios),
        "copy_run_gt_10_rate": mean([float(value > 10) for value in copy_runs]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate natural legal QA predictions with copy and citation metrics.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL with prompt/gold/prediction fields")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(evaluate(Path(args.predictions)), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
