#!/usr/bin/env bash
set -euo pipefail

source ~/.zshrc
mkdir -p outputs/adapters outputs/models
PYTHONPATH=src python3 -m lora_mlx.lora \
  --model mlx_model \
  --train \
  --data data/pasalid \
  --iters 1000 \
  --batch-size 1 \
  --lora-layers 4 \
  --adapter-file outputs/adapters/adapters_pasalid_tinyllama.npz
