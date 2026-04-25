#!/usr/bin/env zsh
set -eo pipefail

source ~/.zshrc
set -u

RUN_NAME=${1:?usage: scripts/query_doc_to_lora_tinyllama.sh <run-name> <question> [max-tokens]}
QUESTION=${2:?usage: scripts/query_doc_to_lora_tinyllama.sh <run-name> <question> [max-tokens]}
MAX_TOKENS=${3:-160}
ADAPTER_FILE="outputs/adapters/doc_to_lora_${RUN_NAME}.npz"
DATA_DIR="data/doc_to_lora/${RUN_NAME}"

PROMPT="Anda adalah model yang sudah menginternalisasi dokumen '${RUN_NAME}' sebagai memori LoRA. Jawab berdasarkan memori dokumen, tanpa meminta konteks tambahan.
Dokumen: ${RUN_NAME}
Q: ${QUESTION}
A: "

PYTHONPATH=src python3 -m lora_mlx.lora \
  --model mlx_model \
  --data "${DATA_DIR}" \
  --adapter-file "${ADAPTER_FILE}" \
  --lora-layers 4 \
  --max-tokens "${MAX_TOKENS}" \
  --temp 0.0 \
  --prompt "${PROMPT}"
