#!/usr/bin/env zsh
set -eo pipefail

source ~/.zshrc
set -u

MODE=${1:-train}
DATASET=${2:-data/doc_to_lora/sakana_gemma_multi_64.jsonl}
CHECKPOINT=${3:-outputs/doc_to_lora/hypernet_gemma_multi64_learned_ce_lr5e5.npz}
MODEL=${4:-mlx-community/gemma-2-2b-it}
ITERS=${5:-24}
LEARNING_RATE=${6:-5e-5}

if [[ "${MODE}" != "train" && "${MODE}" != "eval" ]]; then
  print "usage: $0 [train|eval] [dataset_jsonl] [checkpoint_npz] [model] [iters] [learning_rate]" >&2
  exit 2
fi

CHECKPOINT_ARGS=()
if [[ "${MODE}" == "train" ]]; then
  CHECKPOINT_ARGS=(--save-hypernet "${CHECKPOINT}")
else
  ITERS=0
  CHECKPOINT_ARGS=(--load-hypernet "${CHECKPOINT}")
fi

PYTHONPATH=src python3 scripts/train_doc_to_lora_token_smoke.py \
  --model "${MODEL}" \
  --dataset-jsonl "${DATASET}" \
  --max-examples 64 \
  --eval-examples 16 \
  --iters "${ITERS}" \
  --lora-layers 2 \
  --target-modules down_proj \
  --max-specs 2 \
  --hidden-size 128 \
  --rank 4 \
  --context-encoder model-embed \
  --context-max-tokens 512 \
  --context-chunk-tokens 128 \
  --chunk-merge learned \
  --max-context-chunks 8 \
  --per-rank-gen \
  --per-layer-processing \
  --num-pre-head-layers 1 \
  --loss-scope full-answer \
  --loss-type ce \
  --learning-rate "${LEARNING_RATE}" \
  "${CHECKPOINT_ARGS[@]}"
