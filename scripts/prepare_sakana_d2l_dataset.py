#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DATASET_REPO = "SakanaAI/self_gen_qa_d2l"
DEFAULT_MODEL_PREFIX = "google/gemma-2-2b-it_temp_0.0_closed_qa_prob_1.0"
DEFAULT_DATASETS = ("squad_compact", "pwc_compact", "drop_compact", "ropes_compact")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare SakanaAI Doc-to-LoRA self-generated dataset samples."
    )
    parser.add_argument("--repo", default=DATASET_REPO)
    parser.add_argument("--model-prefix", default=DEFAULT_MODEL_PREFIX)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--split", default="train")
    parser.add_argument("--raw-dir", default="data/raw_datasets/self_gen")
    parser.add_argument("--output", default="data/doc_to_lora/sakana_gemma_sample.jsonl")
    parser.add_argument("--tokenizer", default="mlx-community/gemma-2-2b-it")
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--list-files", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--convert", action="store_true")
    return parser.parse_args()


def selected_files(args) -> list[str]:
    prefixes = [x.strip() for x in args.datasets.split(",") if x.strip()]
    api = HfApi()
    files = api.list_repo_files(args.repo, repo_type="dataset")
    selected = []
    for prefix in prefixes:
        stem = f"{args.model_prefix}/{prefix}/{args.split}/"
        matches = sorted(
            f for f in files if f.startswith(stem) and f.endswith(".parquet")
        )
        selected.extend(matches[: args.max_files])
    return selected


def download_files(args, files: list[str]) -> Path:
    if not files:
        raise ValueError("no files selected for download")
    local_dir = Path(args.raw_dir)
    snapshot_download(
        args.repo,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=files,
    )
    return local_dir


def convert_files(args, files: list[str]) -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The optional 'datasets' package is required for --convert. "
            "Install it with: python3 -m pip install datasets pyarrow"
        ) from exc

    raw_dir = Path(args.raw_dir)
    data_files = [str(raw_dir / f) for f in files if (raw_dir / f).exists()]
    if not data_files:
        raise FileNotFoundError(
            f"No selected parquet files exist under {raw_dir}. Run with --download first."
        )

    from transformers import AutoTokenizer

    ds = load_dataset("parquet", data_files=data_files, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w") as fid:
        for row in ds:
            for example in extract_examples(row, tokenizer):
                fid.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1
                if written >= args.max_examples:
                    return written
    return written


def extract_examples(row: dict, tokenizer=None) -> list[dict]:
    context = row.get("context") or row.get("document")
    prompts = row.get("prompts") or row.get("prompt") or row.get("questions")
    responses = row.get("responses") or row.get("response") or row.get("answers")
    if context is None and "ctx_ids" in row and tokenizer is not None:
        context = tokenizer.decode(row["ctx_ids"], skip_special_tokens=True).strip()
    if prompts is None and responses is None and "input_ids" in row:
        prompts, responses = decode_prompt_response_pairs(row, tokenizer)
    if context is None or prompts is None or responses is None:
        return []
    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(responses, str):
        responses = [responses]
    out = []
    for prompt, response in zip(prompts, responses):
        if not prompt or not response:
            continue
        out.append(
            {
                "document": context,
                "prompt": prompt,
                "response": response,
                "source": "SakanaAI/self_gen_qa_d2l",
            }
        )
    return out


def decode_prompt_response_pairs(row: dict, tokenizer) -> tuple[list[str], list[str]]:
    if tokenizer is None:
        return [], []
    input_ids = row.get("input_ids") or []
    spans = row.get("response_start_end") or []
    prompts = []
    responses = []
    for ids, span in zip(input_ids, spans):
        if not ids or not span or len(span) != 2:
            continue
        start, end = int(span[0]), int(span[1])
        prompt_ids = ids[:start]
        response_ids = ids[start:end]
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True).strip()
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        if prompt and response:
            prompts.append(prompt)
            responses.append(response)
    return prompts, responses


def main():
    args = parse_args()
    files = selected_files(args)
    print(f"selected_files={len(files)}")
    for file in files:
        print(file)

    if args.list_files and not args.download and not args.convert:
        return
    if args.download:
        download_files(args, files)
    if args.convert:
        written = convert_files(args, files)
        print(f"output={args.output}")
        print(f"written={written}")


if __name__ == "__main__":
    main()
