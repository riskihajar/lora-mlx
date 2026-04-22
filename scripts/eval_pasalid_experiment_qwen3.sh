#!/usr/bin/env bash
set -euo pipefail

source ~/.zshrc
PYTHONPATH=src python3 -m lora_mlx.evaluation \
  --model mlx-community/Qwen3-4B-8bit \
  --adapter-file outputs/adapters/adapters_pasalid_qwen3_experiment.npz \
  --lora-layers 4 \
  --data data/pasalid/experiment_split/test_seen.jsonl \
  --max-new-tokens 96 \
  --preview 5 \
  "$@"
