#!/usr/bin/env zsh
set -eo pipefail

source ~/.zshrc
set -u

DATASET=${1:-data/doc_to_lora/sakana_gemma_squad_sample.jsonl}
ITERS=${2:-20}
MAX_EXAMPLES=${3:-5}
EVAL_EXAMPLES=${4:-1}
MODEL=${5:-mlx-community/gemma-2-2b-it}

PYTHONPATH=src python3 scripts/train_doc_to_lora_token_smoke.py \
  --model "${MODEL}" \
  --dataset-jsonl "${DATASET}" \
  --max-examples "${MAX_EXAMPLES}" \
  --eval-examples "${EVAL_EXAMPLES}" \
  --iters "${ITERS}" \
  --lora-layers 2 \
  --target-modules down_proj \
  --max-specs 2 \
  --hidden-size 128 \
  --rank 4 \
  --context-encoder token-hash \
  --context-buckets 8192 \
  --context-latents 8 \
  --loss-scope full-answer
