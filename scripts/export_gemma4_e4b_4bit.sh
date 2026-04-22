#!/usr/bin/env bash
set -euo pipefail

source ~/.zshrc
mkdir -p outputs/predictions
PYTHONPATH=src python3 -m lora_mlx.export \
  --model mlx-community/gemma-4-e4b-it-4bit \
  --adapter-file outputs/adapters/adapters_gemma4_e4b_4bit.npz \
  --lora-layers 4 \
  --max-new-tokens 24 \
  --stop-strings $'\nQ:' $'\nA:' 'table:' 'columns:' \
  --output outputs/predictions/gemma4_e4b_4bit_predictions.jsonl \
  "$@"
