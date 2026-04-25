#!/usr/bin/env zsh
set -eo pipefail

source ~/.zshrc
set -u

DOC_PATH=${1:?usage: scripts/train_doc_to_lora_tinyllama.sh <doc-path> [run-name] [iters]}
RUN_NAME=${2:-$(basename "${DOC_PATH%.*}")}
ITERS=${3:-300}
DATA_DIR="data/doc_to_lora/${RUN_NAME}"
ADAPTER_FILE="outputs/adapters/doc_to_lora_${RUN_NAME}.npz"

mkdir -p outputs/adapters "${DATA_DIR}"

PYTHONPATH=src python3 scripts/build_doc_to_lora_dataset.py \
  --input "${DOC_PATH}" \
  --output-dir "${DATA_DIR}" \
  --title "${RUN_NAME}"

PYTHONPATH=src python3 -m lora_mlx.lora \
  --model mlx_model \
  --train \
  --test \
  --data "${DATA_DIR}" \
  --iters "${ITERS}" \
  --batch-size 1 \
  --lora-layers 4 \
  --learning-rate 1e-5 \
  --adapter-file "${ADAPTER_FILE}"

printf 'data_dir=%s\nadapter_file=%s\n' "${DATA_DIR}" "${ADAPTER_FILE}"
