#!/usr/bin/env zsh
set -eo pipefail

source ~/.zshrc
set -u

mkdir -p outputs/adapters outputs/models
PYTHONPATH=src python3 -m lora_mlx.lora \
  --model mlx_model \
  --train \
  --data data/pasalid/natural_legal_split \
  --iters 1000 \
  --batch-size 1 \
  --lora-layers 4 \
  --adapter-file outputs/adapters/adapters_pasalid_tinyllama_natural_legal.npz
