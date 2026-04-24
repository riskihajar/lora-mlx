#!/usr/bin/env zsh
set -eo pipefail

source ~/.zshrc
set -u
PYTHONPATH=src python3 scripts/export_pasalid_experiment_abc.py \
  --preset tinyllama_final \
  --split seen \
  --experiment-dir data/pasalid/json_final_split \
  --output-dir outputs/predictions/pasalid_experiment_json_final \
  --max-new-tokens 96
PYTHONPATH=src python3 scripts/export_pasalid_experiment_abc.py \
  --preset tinyllama_final \
  --split unseen \
  --experiment-dir data/pasalid/json_final_split \
  --output-dir outputs/predictions/pasalid_experiment_json_final \
  --max-new-tokens 96
PYTHONPATH=src python3 scripts/review_pasalid_experiment_abc.py \
  --preset tinyllama_final \
  --split seen \
  --export-dir outputs/predictions/pasalid_experiment_json_final \
  --summary-only
PYTHONPATH=src python3 scripts/review_pasalid_experiment_abc.py \
  --preset tinyllama_final \
  --split unseen \
  --export-dir outputs/predictions/pasalid_experiment_json_final \
  --summary-only
