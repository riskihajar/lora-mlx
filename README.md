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

## Why LoRA And QLoRA

The main goal of LoRA and QLoRA is to adapt large language models to a new task without paying the cost of full fine-tuning.

- full fine-tuning updates most or all model weights and is expensive in memory and compute
- LoRA keeps the base model frozen and trains only small low-rank adapter weights on selected layers
- QLoRA applies the same idea on top of a quantized base model, which reduces memory use even further

### LoRA

LoRA stands for `Low-Rank Adaptation`.

- instead of updating a large weight matrix directly, LoRA learns a small low-rank update
- in practice, this repo applies adapters to attention projections such as `q_proj` and `v_proj`
- the base model stays unchanged and only the adapter parameters are trained

Main advantages of LoRA:

- much lower memory cost than full fine-tuning
- faster iteration for experiments
- small adapter files that are easy to store and swap
- one base model can support multiple task-specific adapters

Main tradeoffs of LoRA:

- it may not match full fine-tuning on every task
- results depend heavily on adapter target modules, rank, learning rate, and model behavior

### QLoRA

QLoRA applies LoRA on top of a quantized model, typically `4-bit` or `8-bit`.

- the base model is compressed to reduce memory pressure
- LoRA adapters are still trained on top of that quantized base
- this makes it possible to fine-tune larger models on more limited hardware

Main advantages of QLoRA:

- even lower memory usage than standard LoRA
- lets larger models fit into local or smaller experimental environments
- often the most practical option for resource-constrained experimentation

Main tradeoffs of QLoRA:

- more sensitive to implementation details
- quantization can reduce model quality
- some models need extra compatibility work before training and evaluation are reliable

### Practical Summary

- use full fine-tuning when you want maximum flexibility and have the hardware budget
- use LoRA when you want a strong cost/performance tradeoff
- use QLoRA when memory is the main constraint and you still want to adapt a larger model

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
