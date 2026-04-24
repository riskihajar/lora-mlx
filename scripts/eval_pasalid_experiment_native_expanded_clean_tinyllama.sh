#!/usr/bin/env zsh
set -eo pipefail

source ~/.zshrc
set -u

PYTHONPATH=src python3 -m lora_mlx.evaluation \
  --model mlx_model \
  --adapter-file outputs/adapters/adapters_pasalid_tinyllama_native_expanded_clean.npz \
  --lora-layers 4 \
  --data data/pasalid/json_native_expanded_clean_split/test_seen.jsonl \
  --max-new-tokens 128 \
  --preview 5 \
  "$@"
