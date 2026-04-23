#!/usr/bin/env bash
set -euo pipefail

source ~/.zshrc
PYTHONPATH=src python3 -m lora_mlx.evaluation \
  --model mlx_model \
  --adapter-file outputs/adapters/adapters_pasalid_source_tinyllama.npz \
  --lora-layers 4 \
  --data data/pasalid_source/test.jsonl \
  --max-new-tokens 64 \
  --preview 5 \
  "$@"
