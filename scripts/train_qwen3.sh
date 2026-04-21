#!/usr/bin/env bash
set -euo pipefail

source ~/.zshrc
mkdir -p outputs/adapters outputs/models
PYTHONPATH=src python3 -m lora_mlx.lora --model mlx-community/Qwen3-4B-8bit --train --data data --iters 600 --batch-size 1 --lora-layers 4 --adapter-file outputs/adapters/adapters_qwen3_4b_8bit.npz
