#!/usr/bin/env bash
set -euo pipefail

source ~/.zshrc
mkdir -p outputs/adapters outputs/models
PYTHONPATH=src python3 -m lora_mlx.lora \
  --model mlx-community/Qwen3-4B-8bit \
  --train \
  --data data/pasalid_source \
  --iters 150 \
  --batch-size 1 \
  --lora-layers 4 \
  --adapter-file outputs/adapters/adapters_pasalid_source_qwen3.npz
