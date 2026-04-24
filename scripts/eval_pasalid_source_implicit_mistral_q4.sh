#!/usr/bin/env zsh
set -eo pipefail

source ~/.zshrc
set -u
PYTHONPATH=src python3 -m lora_mlx.export \
  --model mlx_model_mistral_q4 \
  --adapter-file outputs/adapters/adapters_pasalid_source_implicit_mistral_q4.npz \
  --data data/pasalid_source/implicit/test.jsonl \
  --lora-layers 4 \
  --max-new-tokens 64 \
  --output outputs/predictions/pasalid_source_implicit_mistral_q4_test.jsonl \
  "$@"
