#!/usr/bin/env bash
set -euo pipefail

source ~/.zshrc
mkdir -p outputs/predictions
PYTHONPATH=src python3 -m lora_mlx.export "$@"
