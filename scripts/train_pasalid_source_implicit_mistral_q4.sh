#!/usr/bin/env zsh
set -eo pipefail

source ~/.zshrc
set -u
mkdir -p outputs/adapters outputs/models
PYTHONPATH=src python3 -m lora_mlx.lora \
  --model mlx_model_mistral_q4 \
  --train \
  --data data/pasalid_source/implicit \
  --iters 150 \
  --batch-size 1 \
  --lora-layers 4 \
  --adapter-file outputs/adapters/adapters_pasalid_source_implicit_mistral_q4.npz
