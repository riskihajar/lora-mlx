#!/usr/bin/env bash
set -euo pipefail

source ~/.zshrc
PYTHONPATH=src python3 -m lora_mlx.export \
  --model mlx_model_mistral_q4 \
  --adapter-file outputs/adapters/adapters_pasalid_source_mistral_q4.npz \
  --data data/pasalid_source/test.jsonl \
  --lora-layers 4 \
  --max-new-tokens 64 \
  --output outputs/predictions/pasalid_source_mistral_q4_test.jsonl \
  "$@"
