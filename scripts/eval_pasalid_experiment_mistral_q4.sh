#!/usr/bin/env bash
set -euo pipefail

source ~/.zshrc
PYTHONPATH=src python3 -m lora_mlx.evaluation \
  --model mlx_model_mistral_q4 \
  --adapter-file outputs/adapters/adapters_pasalid_mistral_q4_experiment.npz \
  --lora-layers 4 \
  --data data/pasalid/experiment_split/test_seen.jsonl \
  --max-new-tokens 96 \
  --preview 5 \
  "$@"
