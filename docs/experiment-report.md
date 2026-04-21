# LoRA Experiment Report

## Context

- Workspace: `lora-mlx`
- Machine: Apple M4, 24 GB RAM
- Dataset: sample WikiSQL data in `data/`
- Goal: reproduce TinyLlama LoRA behavior in the cleaned-up standalone repo before extending model comparisons

## Setup

### Base model

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Local MLX model path: `mlx_model`
- Source: copied from the prior working repo `mlx-examples/lora`

### Training configuration

- Script: `src/lora_mlx/lora.py`
- Data directory: `data`
- Batch size: `1`
- LoRA layers: `4`
- Iterations: `1000`
- Adapter output: `outputs/adapters/adapters_tinyllama.npz`

## What LoRA Changes in This Repo

- The base model is loaded and frozen before training.
- LoRA adapters are injected into the last `4` transformer layers.
- In this setup, LoRA is attached to attention projections `q_proj` and `v_proj`.
- Only adapter parameters are updated during training; the original model weights stay unchanged.

## Qualitative Observation

### Base model behavior

For a WikiSQL prompt such as:

```text
table: 1-10015132-16
columns: Player, No., Nationality, Position, Years in Toronto, School/Club Team
Q: What is terrence ross' nationality
A:
```

The fresh baseline run in this repo still behaves like a general assistant rather than a text-to-SQL model. Example baseline output from the new run:

```text
Terrance Ross is a Canadian professional basketball player who currently plays for the Orlando Magic of the NBA.
```

This shows that the base model understands the question semantically, but does not follow the target SQL output format.

### LoRA behavior after training

After LoRA training in this repo, the model produces outputs that are often structurally close to SQL. Example from the new run:

```text
SELECT Nationality FROM 1-10015132-16 WHERE Player = 'Terrance Ross'
```

This still misses exact match because of a small entity typo (`Terrance` vs `Terrence`), but it shows that the adapter shifts the model toward the target text-to-SQL format.

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
| TinyLlama + LoRA | 1000 | 1.676 | 5.343 | 0.0000 | 0.3135 |

## Training Notes

- Initial validation loss at iteration `1`: `2.544`
- Validation loss at iteration `600`: `1.443`
- Final validation loss at iteration `1000`: `1.417`
- Training completed successfully and saved the final adapter to `outputs/adapters/adapters_tinyllama.npz`

## Evaluation Notes

- Test perplexity was computed on `100` test batches using `src/lora_mlx/lora.py`
- Baseline EM/F1 was computed without any adapter over the same `100` test examples
- EM/F1 were computed on the full `100` examples in `data/test.jsonl`
- Baseline predictions were exported to `outputs/predictions/tinyllama_base_test_predictions.jsonl`
- Predictions were exported to `outputs/predictions/tinyllama_1000_test_predictions.jsonl`

## Interpretation

- The standalone repo reproduces the expected TinyLlama LoRA workflow successfully.
- The baseline TinyLlama result confirms that the untuned model mostly answers in natural language or drifts off-format.
- Perplexity is very close to the earlier experiment, which is a good sign that the refactored repo preserves the training path.
- Moving from baseline `F1 = 0.1089` to LoRA `F1 = 0.3135` is a meaningful gain in structural task alignment.
- `F1 = 0.3135` shows the model often captures useful SQL structure even when exact match remains `0.0000`.
- The largest remaining failure modes appear to be:
  - entity spelling mismatches
  - wrong target column selection
  - prompt continuation or format drift
  - partially correct SQL with one fatal field error

## Comparison With Earlier Repo

Reference result from the previous `mlx-examples/lora` experiment for TinyLlama `1000`:

- previous run: `Test Loss = 1.674`, `PPL = 5.333`, `EM = 0.0300`, `F1 = 0.3245`
- current baseline: `EM = 0.0000`, `F1 = 0.1089`
- current LoRA run: `Test Loss = 1.676`, `PPL = 5.343`, `EM = 0.0000`, `F1 = 0.3135`

This is close enough to treat the new repo as a working continuation of the earlier TinyLlama setup.

## Artifacts

- Local TinyLlama MLX model: `mlx_model`
- Dataset splits:
  - `data/train.jsonl`
  - `data/valid.jsonl`
  - `data/test.jsonl`
- Final adapter: `outputs/adapters/adapters_tinyllama.npz`
- Exported baseline predictions: `outputs/predictions/tinyllama_base_test_predictions.jsonl`
- Exported predictions: `outputs/predictions/tinyllama_1000_test_predictions.jsonl`

## Recommended Next Step

Build the same report sections for the stronger models so the final comparison table can include:

1. `TinyLlama`
2. `Mistral q4`
3. `Qwen3 4B 8bit`
4. `Gemma 4 e4b 4bit` if we choose to benchmark it on the same task
