# Standalone Migration Plan

## Goal

Extract the LoRA/QLoRA experiment work from `mlx-examples/lora` into a dedicated repository:

- target repo: `git@github.com:riskihajar/lora-mlx.git`

The standalone repo should separate:

- source code
- experiment utilities
- documentation
- generated artifacts
- optional local-only outputs

## Current State

Current work is mixed inside the example directory and contains both source files and generated artifacts.

### Source-like files

- `convert.py`
- `fuse.py`
- `lora.py`
- `models.py`
- `utils.py`
- `requirements.txt`
- `evaluate_em_f1.py`
- `export_predictions.py`

### Documentation files

- `lora_experiment_report.md`
- `technical_notes.md`
- `mistral_q4_1000_error_analysis.md`

### Generated artifacts

- adapter checkpoints: `adapters_*.npz`
- converted model folders: `mlx_model/`, `mlx_model_mistral_q4/`
- prediction exports: `*_test_predictions.jsonl`

## Proposed Repository Structure

```text
lora-mlx/
  README.md
  requirements.txt
  .gitignore
  src/
    lora_mlx/
      __init__.py
      convert.py
      fuse.py
      lora.py
      models.py
      utils.py
      evaluation.py
      export.py
  scripts/
    train_tinyllama.sh
    train_mistral_q4.sh
    train_qwen3.sh
    eval_em_f1.sh
    export_predictions.sh
  configs/
    tinyllama.yaml
    mistral_q4.yaml
    qwen3_4b_8bit.yaml
  docs/
    experiment-report.md
    technical-notes.md
    mistral-q4-error-analysis.md
    migration-notes.md
  examples/
    prompts/
      wikisql_prompt.txt
  data/
    .gitkeep
  outputs/
    adapters/
    predictions/
    reports/
    models/
```

## Structure Rationale

### `src/lora_mlx/`

Holds reusable Python code.

Recommended file mapping:

- `convert.py` -> conversion entrypoint
- `fuse.py` -> adapter fusion entrypoint
- `lora.py` -> train/test/generate entrypoint
- `models.py` -> model definitions
- `utils.py` -> model loading and shared helpers
- `evaluation.py` -> logic from `evaluate_em_f1.py`
- `export.py` -> logic from `export_predictions.py`

### `scripts/`

Holds runnable shell wrappers for common tasks so commands are easier to reproduce.

### `configs/`

Stores experiment presets instead of burying configuration in long CLI commands.

### `docs/`

Stores all markdown notes and reports.

### `outputs/`

Stores generated files that should usually stay out of git.

Suggested conventions:

- `outputs/adapters/`
- `outputs/predictions/`
- `outputs/reports/`
- `outputs/models/`

## What Should Be Committed

Recommended to commit:

- all Python source under `src/`
- shell scripts under `scripts/`
- config presets under `configs/`
- Markdown docs under `docs/`
- a small sample prompt or sample config
- optionally a tiny sample dataset format example

Recommended not to commit:

- `mlx_model/`
- `mlx_model_mistral_q4/`
- any large downloaded model directory
- `adapters_*.npz`
- `*_test_predictions.jsonl`
- cache files
- Hugging Face downloads

## Suggested `.gitignore`

```gitignore
__pycache__/
.DS_Store
.pytest_cache/
.venv/

outputs/
data/

*.npz
*.safetensors
*.jsonl

mlx_model/
mlx_model_*/
```

If you want to keep selected JSONL reports in git, move them into `docs/` or `reports/` explicitly instead of tracking everything broadly.

## Migration Steps

### Phase 1: bootstrap standalone repo

1. clone `git@github.com:riskihajar/lora-mlx.git`
2. create base directories
3. add `README.md`, `requirements.txt`, `.gitignore`

### Phase 2: move code

1. copy source files into `src/lora_mlx/`
2. keep current filenames first to minimize breakage
3. fix imports from relative flat-file imports to package imports

### Phase 3: move utilities

1. move `evaluate_em_f1.py` logic into `src/lora_mlx/evaluation.py`
2. move `export_predictions.py` logic into `src/lora_mlx/export.py`
3. keep thin CLI wrappers if desired

### Phase 4: move documentation

1. move `lora_experiment_report.md` -> `docs/experiment-report.md`
2. move `technical_notes.md` -> `docs/technical-notes.md`
3. move `mistral_q4_1000_error_analysis.md` -> `docs/mistral-q4-error-analysis.md`

### Phase 5: define output conventions

1. place adapter outputs under `outputs/adapters/`
2. place prediction exports under `outputs/predictions/`
3. place converted models under `outputs/models/`

### Phase 6: reproducibility cleanup

1. add shell scripts for standard runs
2. add config presets for TinyLlama, Mistral q4, and Qwen3
3. document which metrics use full test set vs partial batches

## Important Pain Points to Preserve in the New Repo

### 1. Qwen3 required local compatibility patches

Do not lose the `models.py` changes that enabled:

- `head_dim`
- `q_norm`
- `k_norm`
- tied embeddings
- quantized embedding output projection

Without these, Qwen3 loading will fail again.

### 2. Current code still reflects example-repo assumptions

The implementation still inherits architectural assumptions from the original Apple example. Even after patching Qwen3, it is not yet a general-purpose backend for every model family.

### 3. Evaluation settings are not uniform across all runs

Some Qwen3 perplexity runs used fewer test batches due to runtime limits. This must be documented clearly in the standalone repo to avoid unfair comparisons.

### 4. `lora.py` prompt mode depends on adapter file existence

This is awkward for base-only generation. In the standalone repo, consider loosening that requirement.

### 5. Generated artifacts can quickly pollute the repo

The new repo should aggressively separate code/docs from outputs.

## Recommended First Implementation Scope

For the first standalone version, keep it conservative:

- copy current working source
- keep CLI behavior mostly unchanged
- improve layout and documentation first
- avoid large refactors until after the repo is stable

## Immediate Next Step

When the local `lora-mlx` repo exists on disk, start by creating this structure:

```text
README.md
.gitignore
requirements.txt
src/lora_mlx/
docs/
scripts/
configs/
outputs/
```

Then migrate code before moving artifacts.
