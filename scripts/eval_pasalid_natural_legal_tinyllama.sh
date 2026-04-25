#!/usr/bin/env zsh
set -eo pipefail

source ~/.zshrc
set -u

SPLIT=${1:-seen}
OUTPUT_DIR=${2:-outputs/predictions/pasalid_natural_legal}

PYTHONPATH=src python3 scripts/export_pasalid_experiment_abc.py \
  --preset tinyllama_natural_legal \
  --experiment-dir data/pasalid/natural_legal_split \
  --output-dir "$OUTPUT_DIR" \
  --split "$SPLIT" \
  --include-d \
  --max-new-tokens 160

PYTHONPATH=src python3 scripts/review_pasalid_experiment_abc.py \
  --preset tinyllama_natural_legal \
  --split "$SPLIT" \
  --export-dir "$OUTPUT_DIR" \
  --summary-only

for condition in A_base_no_context B_base_with_context C_adapter_no_context D_adapter_with_context; do
  PYTHONPATH=src python3 scripts/eval_pasalid_natural_metrics.py \
    --predictions "$OUTPUT_DIR/tinyllama_natural_legal_${SPLIT}_${condition}.jsonl"
done
