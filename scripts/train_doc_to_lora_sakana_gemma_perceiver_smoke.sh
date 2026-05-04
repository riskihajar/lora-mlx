#!/usr/bin/env zsh
set -eo pipefail

source ~/.zshrc
set -u

DATASET=${1:-data/doc_to_lora/sakana_gemma_multi_64.jsonl}
CHECKPOINT=${2:-outputs/doc_to_lora/hypernet_gemma_perceiver_smoke.npz}
MODEL=${3:-mlx-community/gemma-2-2b-it}
ITERS=${4:-1}
BEST_CHECKPOINT=${5:-${CHECKPOINT:r}_best.npz}

PYTHONPATH=src python3 scripts/train_doc_to_lora_token_smoke.py \
  --model "${MODEL}" \
  --dataset-jsonl "${DATASET}" \
  --max-examples 8 \
  --eval-examples 2 \
  --iters "${ITERS}" \
  --eval-every 1 \
  --batch-size 2 \
  --lora-layers 1 \
  --target-modules down_proj \
  --max-specs 1 \
  --hidden-size 64 \
  --rank 2 \
  --context-encoder model-embed \
  --hypernet-aggregator perceiver \
  --context-max-tokens 128 \
  --context-chunk-tokens 0 \
  --perceiver-latents 8 \
  --perceiver-blocks 1 \
  --perceiver-self-attn 1 \
  --loss-scope full-answer \
  --loss-type ce \
  --learning-rate 5e-5 \
  --save-hypernet "${CHECKPOINT}" \
  --save-best-hypernet "${BEST_CHECKPOINT}"
