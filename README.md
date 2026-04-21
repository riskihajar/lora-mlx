# lora-mlx

Standalone playground for LoRA/QLoRA experiments on MLX.

## Scope

This repository extracts and extends work originally explored inside `mlx-examples/lora`.

It focuses on:

- lightweight LoRA/QLoRA experiments
- MLX-compatible model loading
- structured evaluation with `PPL`, `EM`, and `F1`
- experiment documentation and reproducibility

## Layout

```text
src/lora_mlx/      core source files
scripts/           reproducible shell entrypoints
configs/           experiment presets
docs/              reports, technical notes, migration notes
examples/          prompts and small examples
outputs/           generated artifacts (models, adapters, predictions)
```

## Current Focus

- TinyLlama baseline experiments
- Mistral q4 QLoRA experiments
- Qwen3 compatibility patching and experiments
- Gemma 4 text-only compatibility work

## Notes

- Large model files and generated artifacts are intended to stay outside git tracking.
- See `docs/technical-notes.md` for important caveats and pain points.

## Run Modules

Use package-style execution from the repo root:

```bash
source ~/.zshrc && PYTHONPATH=src python3 -m lora_mlx.lora --help
source ~/.zshrc && PYTHONPATH=src python3 -m lora_mlx.convert --help
source ~/.zshrc && PYTHONPATH=src python3 -m lora_mlx.evaluation --help
source ~/.zshrc && PYTHONPATH=src python3 -m lora_mlx.export --help
```

## Default Paths

- data: `data/`
- adapters: `outputs/adapters/`
- predictions: `outputs/predictions/`
- converted models: `outputs/models/`

## Output Convention

- training checkpoints should go into `outputs/adapters/`
- prediction exports should go into `outputs/predictions/`
- converted or fused MLX models should go into `outputs/models/`
- long-form analysis documents should stay in `docs/`
