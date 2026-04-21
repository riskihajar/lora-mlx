#!/usr/bin/env bash
set -euo pipefail

source ~/.zshrc
mkdir -p outputs/adapters outputs/models
PYTHONPATH=src python3 -m lora_mlx.lora --model mlx-community/gemma-4-e4b-it-4bit --train --data data --iters 300 --batch-size 1 --lora-layers 4 --adapter-file outputs/adapters/adapters_gemma4_e4b_4bit.npz
