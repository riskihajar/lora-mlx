#!/usr/bin/env zsh
set -eo pipefail

source ~/.zshrc
set -u

MODEL=${1:-mlx-community/gemma-2-2b-it}
ITERS=${2:-3}

PYTHONPATH=src python3 scripts/train_doc_to_lora_token_smoke.py \
  --model "${MODEL}" \
  --iters "${ITERS}" \
  --num-docs 1 \
  --lora-layers 1 \
  --target-modules down_proj \
  --max-specs 1 \
  --hidden-size 32 \
  --rank 1
