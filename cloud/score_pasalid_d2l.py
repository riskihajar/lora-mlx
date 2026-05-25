"""Score Sakana D2L Pasal.id cloud predictions.

Reads ``predictions_<split>.jsonl`` files emitted by
``cloud/eval_pasalid_d2l.sh``, parses both expected and generated answers
(stripping the chat template tags from the model output, extracting the
``answer`` field from the gold JSON envelope), and reports token-level F1 and
EM plus simple length / latency statistics per split.

Usage:

    python3 cloud/score_pasalid_d2l.py
    python3 cloud/score_pasalid_d2l.py --run full_177
    python3 cloud/score_pasalid_d2l.py --pred-dir path/to/run_dir
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = REPO_ROOT / "outputs/predictions/pasalid_d2l_cloud"


def _normalize_tokens(text: str) -> list[str]:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([(),.:;\"\'\\[\\]{}])\s*", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().split()


def _token_f1(prediction: str, gold: str) -> float:
    from collections import Counter

    pred_tokens = _normalize_tokens(prediction)
    gold_tokens = _normalize_tokens(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, gold: str) -> int:
    return int(_normalize_tokens(prediction) == _normalize_tokens(gold))


def _extract_gold_answer(expected: str) -> tuple[str, str]:
    """Return ``(answer_text, source_reference_text)`` from the gold envelope."""
    expected = expected.strip()
    try:
        obj = json.loads(expected)
    except json.JSONDecodeError:
        return expected, ""
    if not isinstance(obj, dict):
        return expected, ""
    answer = str(obj.get("answer", "")).strip()
    parts = [
        str(obj.get("source_type", "")).strip(),
        str(obj.get("source_number", "")).strip(),
        str(obj.get("source_year", "")).strip(),
        str(obj.get("source_article", "")).strip(),
    ]
    source = " ".join(p for p in parts if p)
    return answer, source


_MODEL_TURN_RE = re.compile(r"<start_of_turn>model\s*(.*?)(?:<end_of_turn>|$)", re.S)


def _strip_model_response(generated: str) -> str:
    """Pull the assistant turn out of the Gemma chat-template output.

    Falls back to the raw string if the expected markers are missing.
    """
    matches = _MODEL_TURN_RE.findall(generated)
    if not matches:
        return generated.strip()
    return matches[-1].strip()


def _format_source_reference(record: dict) -> str:
    return (record.get("source_reference") or "").strip()


def _stats(values: Iterable[float]) -> dict[str, float]:
    vals = list(values)
    if not vals:
        return {"n": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }


def _score_split(pred_path: Path) -> dict:
    with pred_path.open() as fp:
        rows = [json.loads(line) for line in fp]
    if not rows:
        return {}

    f1s: list[float] = []
    ems: list[int] = []
    src_token_match: list[int] = []
    gen_lens: list[int] = []
    gen_lat: list[float] = []
    int_lat: list[float] = []
    sample_records: list[dict] = []

    for r in rows:
        gold_answer, gold_source = _extract_gold_answer(r["expected_answer"])
        pred_text = _strip_model_response(r["generated"])

        f1 = _token_f1(pred_text, gold_answer)
        em = _exact_match(pred_text, gold_answer)

        # very loose source match: does the generated text mention the source
        # reference token-by-token (e.g. "uu", "1", "2024", "27a")?
        ref_text = _format_source_reference(r) or gold_source
        ref_tokens = set(_normalize_tokens(ref_text))
        pred_tokens = set(_normalize_tokens(pred_text))
        ref_overlap = (
            len(ref_tokens & pred_tokens) / len(ref_tokens) if ref_tokens else 0.0
        )

        f1s.append(f1)
        ems.append(em)
        src_token_match.append(ref_overlap)
        gen_lens.append(len(pred_text))
        gen_lat.append(float(r.get("latency_generate_s", 0.0)))
        int_lat.append(float(r.get("latency_internalize_s", 0.0)))

        if len(sample_records) < 5:
            sample_records.append(
                {
                    "qa_id": r["qa_id"],
                    "source_reference": ref_text,
                    "question": r["question"],
                    "gold_answer": gold_answer,
                    "pred_answer": pred_text,
                    "f1": round(f1, 4),
                    "em": em,
                    "src_overlap": round(ref_overlap, 3),
                }
            )

    return {
        "n": len(rows),
        "f1": _stats(f1s),
        "em": _stats(ems),
        "source_overlap": _stats(src_token_match),
        "pred_chars": _stats(gen_lens),
        "latency_generate_s": _stats(gen_lat),
        "latency_internalize_s": _stats(int_lat),
        "samples": sample_records,
    }


def _resolve_run_dir(base: Path, run: str) -> Path:
    candidates = sorted(p for p in base.glob("*/run/*") if p.is_dir())
    if not candidates:
        raise SystemExit(f"[score] no run dirs under {base}")
    if run == "latest":
        return candidates[-1]
    for c in candidates:
        if c.name == run:
            return c
    raise SystemExit(f"[score] run {run!r} not found under {base}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default="latest",
        help="run tag to score (default: latest under outputs/predictions/...)",
    )
    parser.add_argument(
        "--pred-dir",
        default=None,
        help="override: path to a directory containing predictions_*.jsonl",
    )
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    args = parser.parse_args()

    if args.pred_dir is not None:
        run_dir = Path(args.pred_dir)
    else:
        run_dir = _resolve_run_dir(Path(args.base), args.run)
    print(f"[score] run dir: {run_dir}")

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        print(f"[score] cloud summary: {summary_path.read_text()}")

    results: dict[str, dict] = {}
    for pred_path in sorted(run_dir.glob("predictions_*.jsonl")):
        split = pred_path.stem.replace("predictions_", "")
        print(f"\n=== split: {split} ({pred_path.name}) ===")
        s = _score_split(pred_path)
        results[split] = s
        if not s:
            print("  (empty predictions file)")
            continue
        print(
            f"  rows: {s['n']}  "
            f"F1 mean={s['f1']['mean']:.4f}  "
            f"EM mean={s['em']['mean']:.4f}  "
            f"source_overlap mean={s['source_overlap']['mean']:.4f}"
        )
        print(
            f"  pred_chars median={s['pred_chars']['median']:.0f}  "
            f"latency_generate_s median={s['latency_generate_s']['median']:.2f}  "
            f"latency_internalize_s median={s['latency_internalize_s']['median']:.2f}"
        )

    out_path = run_dir / "scoring.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[score] wrote: {out_path}")


if __name__ == "__main__":
    main()
