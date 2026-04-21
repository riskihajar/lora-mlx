#!/usr/bin/env bash
set -euo pipefail

source ~/.zshrc
PYTHONPATH=src python3 -m lora_mlx.lora --model mlx_model --train --data data --iters 600 --batch-size 1 --lora-layers 4 --adapter-file outputs/adapters/adapters_tinyllama.npz
