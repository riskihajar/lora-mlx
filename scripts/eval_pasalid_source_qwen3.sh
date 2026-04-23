#!/usr/bin/env bash
set -euo pipefail

source ~/.zshrc
PYTHONPATH=src python3 -m lora_mlx.export \
  --model mlx-community/Qwen3-4B-8bit \
  --adapter-file outputs/adapters/adapters_pasalid_source_qwen3.npz \
  --data data/pasalid_source/test.jsonl \
  --lora-layers 4 \
  --max-new-tokens 64 \
  --output outputs/predictions/pasalid_source_qwen3_test.jsonl \
  "$@"
