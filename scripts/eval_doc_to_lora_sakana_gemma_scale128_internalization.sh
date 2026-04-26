#!/usr/bin/env zsh
set -eo pipefail

source ~/.zshrc
set -u

SPLIT=${1:-heldout}
DATASET=${2:-data/doc_to_lora/sakana_gemma_multi_256.jsonl}
CHECKPOINT=${3:-outputs/doc_to_lora/hypernet_gemma_multi128_learned_ce_lr5e5_best.npz}
MODEL=${4:-mlx-community/gemma-2-2b-it}

if [[ "${SPLIT}" != "heldout" && "${SPLIT}" != "full" ]]; then
  print "usage: $0 [heldout|full] [dataset_jsonl] [checkpoint_npz] [model]" >&2
  exit 2
fi

SKIP_EXAMPLES=96
MAX_EXAMPLES=32
if [[ "${SPLIT}" == "full" ]]; then
  SKIP_EXAMPLES=0
  MAX_EXAMPLES=128
fi

PYTHONPATH=src python3 scripts/eval_doc_to_lora_internalization.py \
  --model "${MODEL}" \
  --hypernet "${CHECKPOINT}" \
  --dataset-jsonl "${DATASET}" \
  --skip-examples "${SKIP_EXAMPLES}" \
  --max-examples "${MAX_EXAMPLES}" \
  --hidden-size 128 \
  --rank 4 \
  --lora-layers 2 \
  --target-modules down_proj \
  --max-specs 2 \
  --context-max-tokens 512 \
  --context-chunk-tokens 128 \
  --chunk-merge learned \
  --max-context-chunks 8 \
  --per-rank-gen \
  --per-layer-processing \
  --num-pre-head-layers 1
