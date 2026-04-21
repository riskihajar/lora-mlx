# LoRA Experiment Report

## Context

- Workspace: `mlx-examples/lora`
- Machine: Apple M4, 24 GB RAM
- Dataset: sample WikiSQL data in `data/`
- Goal: understand how LoRA changes model behavior compared with the base model

## Setup

### Base model

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Converted local MLX model: `mlx_model`
- Approximate size after convert: `2.1G`

### Stronger quantized model

- Model: `mistralai/Mistral-7B-v0.1`
- Converted local MLX model: `mlx_model_mistral_q4`
- Quantization: 4-bit, group size 64
- Approximate size after convert: `3.8G`

### Training configuration

- Script: `lora.py`
- Data directory: `data`
- Batch size: `1`
- LoRA layers: `4`
- Adapter checkpoints:
  - `adapters_tinyllama_100.npz`
  - `adapters_tinyllama_600.npz`
  - `adapters_tinyllama_1000.npz`
  - `adapters_mistral_q4_100.npz`
  - `adapters_mistral_q4_600.npz`
  - `adapters_mistral_q4_1000.npz`
  - `adapters_qwen3_4b_8bit_100.npz`
  - `adapters_qwen3_4b_8bit_600.npz`
  - `adapters_qwen3_4b_8bit_1000.npz`

## What LoRA Changes in This Repo

- The base model is loaded and then frozen.
- LoRA adapters are injected into the last `N` transformer layers.
- In this example, LoRA is attached to attention projections `q_proj` and `v_proj`.
- Only the small adapter parameters are trained; the original model weights stay unchanged.

## Qualitative Observation

### Base model behavior

For a WikiSQL prompt such as:

```text
table: 1-10015132-16
columns: Player, No., Nationality, Position, Years in Toronto, School/Club Team
Q: What is terrence ross' nationality
A:
```

The base TinyLlama model answered in natural language rather than SQL:

```text
Terrance Ross is a Canadian professional basketball player who currently plays for the Orlando Magic of the NBA.
```

This shows that the base model understands the question semantically, but does not follow the target text-to-SQL format.

### LoRA behavior after training

After LoRA training, the model started generating outputs that resemble SQL much more closely, for example:

```text
SELECT Nationality FROM 1-10015132-16 WHERE Player = 'Terrance Ross'
```

This prediction is not an exact match because of a typo (`Terrance` vs `Terrence`), but it shows that LoRA successfully pushes the model toward the target output structure.

## Metrics

### Metric definition

- `EM`: exact string match after simple whitespace normalization
- `F1`: token-level F1 after lowercase conversion and light SQL token normalization
- `PPL`: perplexity reported by `lora.py --test`

These are lexical metrics, not SQL execution accuracy.

### Results summary

| Model | Iterations | Test Loss | Test PPL | EM | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base TinyLlama | 0 | N/A | N/A | 0.0000 | 0.1089 |
| TinyLlama + LoRA | 100 | 2.029 | 7.604 | 0.0000 | 0.1391 |
| TinyLlama + LoRA | 600 | 1.712 | 5.543 | 0.0200 | 0.3244 |
| TinyLlama + LoRA | 1000 | 1.674 | 5.333 | 0.0300 | 0.3245 |
| Base Mistral q4 | 0 | N/A | N/A | 0.0000 | 0.1394 |
| Mistral q4 + QLoRA | 100 | 1.665 | 5.288 | 0.0400 | 0.2580 |
| Mistral q4 + QLoRA | 600 | 1.551 | 4.716 | 0.2200 | 0.6910 |
| Mistral q4 + QLoRA | 1000 | 1.549 | 4.705 | 0.2400 | 0.7231 |
| Base Qwen3 4B 8bit | 0 | N/A | N/A | 0.0000 | 0.1561 |
| Qwen3 4B 8bit + QLoRA | 100 | 1.598* | 4.946* | 0.0400 | 0.5115 |
| Qwen3 4B 8bit + QLoRA | 600 | 1.242* | 3.464* | 0.1500 | 0.6879 |
| Qwen3 4B 8bit + QLoRA | 1000 | 1.174* | 3.234* | 0.1900 | 0.6109 |

## Interpretation

- LoRA clearly improves the model over the base TinyLlama on this task.
- The biggest gain happens between `100` and `600` iterations.
- From `600` to `1000` iterations, perplexity still improves a bit, but `F1` is almost flat.
- This suggests diminishing returns for the current setup.

## Mistral q4 Findings

- The stronger quantized base model changes the outcome significantly.
- Even after only `100` QLoRA steps, Mistral q4 already beats TinyLlama `1000` on exact match.
- At `600` steps, Mistral q4 produces a large jump in task quality:
  - `EM = 0.2200`
  - `F1 = 0.6910`
  - `PPL = 4.716`
- At `1000` steps, Mistral q4 improves a bit further:
  - `EM = 0.2400`
  - `F1 = 0.7231`
  - `PPL = 4.705`
- This is a much stronger result than TinyLlama `1000`:
  - TinyLlama `1000`: `EM = 0.0300`, `F1 = 0.3245`, `PPL = 5.333`
  - Mistral q4 `1000`: `EM = 0.2400`, `F1 = 0.7231`, `PPL = 4.705`

## Qwen3 Findings

- `Qwen3` required local compatibility patches before it could be used in this repo.
- After patching the lightweight MLX model implementation, `mlx-community/Qwen3-4B-8bit` became usable for loading, generation, and QLoRA training.
- Qwen3 shows a very strong early gain after only `100` steps:
  - `EM = 0.0400`
  - `F1 = 0.5115`
- At `600` steps, it becomes highly competitive with Mistral q4:
  - `EM = 0.1500`
  - `F1 = 0.6879`
  - `PPL = 3.464`
- At `1000` steps, exact match improves, but F1 drops somewhat:
  - `EM = 0.1900`
  - `F1 = 0.6109`
  - `PPL = 3.234`
- This suggests Qwen3 has a different training profile from Mistral q4, with a possible sweet spot around `600` steps for balanced quality.

## Notes on Qwen3 Evaluation

- Qwen3 metrics were measured using `mlx-community/Qwen3-4B-8bit`, not a local HF-to-MLX conversion.
- Qwen3 `PPL` values marked with `*` were computed on `20` test batches for runtime reasons.
- Qwen3 `EM/F1` were computed over the full `100` test examples with `max-new-tokens=32`.

## Why the Gap Matters

- TinyLlama is useful for understanding the LoRA workflow.
- Mistral q4 shows the practical benefit of combining a stronger base model with QLoRA.
- The experiment suggests that model capacity matters at least as much as training duration.
- For strict text-to-SQL tasks, a stronger base model sharply increases the chance of exact SQL matches.
- The gain from `600` to `1000` is real but smaller, which suggests the run is approaching diminishing returns.
- Qwen3 demonstrates that a smaller but newer model can still be highly competitive once compatibility issues are resolved.

## Key Findings

1. TinyLlama base model prefers natural-language answers instead of SQL.
2. LoRA successfully teaches the model the target SQL-like structure.
3. Exact match remains low because the task is strict and errors are often small but fatal:
   - entity typos
   - wrong column selection
   - wrong predicate field
   - partial prompt continuation
4. For learning how LoRA works, this experiment is already successful.
5. For stronger task performance, a stronger base model is needed.
6. Moving from TinyLlama to Mistral q4 produces the biggest improvement in the whole experiment.
7. Qwen3 4B is a strong compact contender and may offer a better quality/speed trade-off than expected.

## Artifacts

- Local TinyLlama MLX model: `mlx_model`
- 100-step adapter: `adapters_tinyllama_100.npz`
- 600-step adapter: `adapters_tinyllama_600.npz`
- 1000-step adapter: `adapters_tinyllama_1000.npz`
- Local Mistral q4 MLX model: `mlx_model_mistral_q4`
- 100-step Mistral q4 adapter: `adapters_mistral_q4_100.npz`
- 600-step Mistral q4 adapter: `adapters_mistral_q4_600.npz`
- 1000-step Mistral q4 adapter: `adapters_mistral_q4_1000.npz`
- Qwen3 MLX community model: `mlx-community/Qwen3-4B-8bit`
- 100-step Qwen3 adapter: `adapters_qwen3_4b_8bit_100.npz`
- 600-step Qwen3 adapter: `adapters_qwen3_4b_8bit_600.npz`
- 1000-step Qwen3 adapter: `adapters_qwen3_4b_8bit_1000.npz`

## Recommended Next Step

Use both Mistral q4 and Qwen3 for deeper analysis:

1. export full test predictions for Qwen3 checkpoints
2. compare Qwen3 vs Mistral error patterns
3. try small hyperparameter variations if needed
