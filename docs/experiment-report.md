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

### Stronger quantized model

- Model: `mistralai/Mistral-7B-v0.1`
- Local MLX model path: `mlx_model_mistral_q4`
- Quantization: 4-bit, group size 64
- Source: copied from the prior working repo `mlx-examples/lora`

### MLX community model

- Model: `mlx-community/Qwen3-4B-8bit`
- Loaded directly from Hugging Face MLX community
- Quantization: 8-bit MLX community release

### Gemma 4 text-only support target

- Model: `mlx-community/gemma-4-e4b-it-4bit`
- Loaded directly from Hugging Face MLX community
- Quantization: 4-bit MLX community release
- Status: text-only loading, training, and perplexity evaluation work in this repo; full generative evaluation still times out

### Training configuration

- Script: `src/lora_mlx/lora.py`
- Data directory: `data`
- Batch size: `1`
- LoRA layers: `4`
- Iterations: `1000`
- Adapter output: `outputs/adapters/adapters_tinyllama.npz`
- Mistral q4 adapter output: `outputs/adapters/adapters_mistral_q4.npz`
- Qwen3 adapter output: `outputs/adapters/adapters_qwen3_4b_8bit.npz`

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
| Base Mistral q4 | 0 | N/A | N/A | 0.0000 | 0.1394 |
| Base Qwen3 4B 8bit | 0 | N/A | N/A | 0.0000 | 0.1561 |
| TinyLlama + LoRA | 1000 | 1.676 | 5.343 | 0.0000 | 0.3135 |
| Mistral q4 + QLoRA | 1000 | 1.561 | 4.765 | 0.1600 | 0.4467 |
| Qwen3 4B 8bit + QLoRA | 1000 | 1.217* | 3.376* | 0.1400 | 0.6946 |
| Base Gemma 4 e4b 4bit | 0 | N/A | N/A | 0.0000 | 0.1065 |
| Gemma 4 e4b 4bit + LoRA | 1000 | 4.670** | 106.661** | 0.0000 | 0.1061 |

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
- Mistral q4 baseline predictions were exported to `outputs/predictions/mistral_q4_base_test_predictions.jsonl`
- Mistral q4 predictions were exported to `outputs/predictions/mistral_q4_1000_test_predictions.jsonl`
- Qwen3 baseline predictions were exported to `outputs/predictions/qwen3_4b_8bit_base_test_predictions.jsonl`
- Qwen3 predictions were exported to `outputs/predictions/qwen3_4b_8bit_1000_test_predictions.jsonl`
- Qwen3 perplexity was computed on `20` test batches for runtime reasons
- Gemma 4 test perplexity was computed on `10` test batches for runtime reasons
- Gemma 4 generation path was later debugged and made consistent for cached vs non-cached decoding on small tests
- Gemma 4 full EM/F1 evaluation now runs on the full `100` examples, but output quality remains poor

## Interpretation

- The standalone repo reproduces the expected TinyLlama LoRA workflow successfully.
- The baseline TinyLlama result confirms that the untuned model mostly answers in natural language or drifts off-format.
- Perplexity is very close to the earlier experiment, which is a good sign that the refactored repo preserves the training path.
- Moving from baseline `F1 = 0.1089` to LoRA `F1 = 0.3135` is a meaningful gain in structural task alignment.
- `F1 = 0.3135` shows the model often captures useful SQL structure even when exact match remains `0.0000`.
- Mistral q4 is clearly stronger than TinyLlama on this task in the new repo as well.
- `EM = 0.1600` and `F1 = 0.4467` show that Mistral q4 generates exact SQL more often and reaches better structural overlap.
- Mistral q4 still shows some prompt continuation drift, so the gap is not purely model strength; decoding behavior also matters.
- Qwen3 has the strongest structural score in the current repo run.
- `F1 = 0.6946` and `PPL = 3.376` suggest Qwen3 is the best current model for near-correct SQL generation on this dataset.
- Qwen3 does not lead Mistral q4 on exact match in this run (`0.1400` vs `0.1600`), which suggests it is often close but not always exact.
- Across base models, `Qwen3` also starts from the strongest baseline F1.
- Gemma 4 support is now functionally integrated for text-only LoRA workflows, including a repaired generation path for full generative evaluation.
- `Test PPL = 106.661`, base `F1 = 0.1065`, and adapted `F1 = 0.1061` suggest this configuration is not competitive with the other three models on this dataset.
- After the decode-path fix, Gemma 4 no longer appears blocked by a broken cache path; it now appears limited mainly by poor task alignment on this WikiSQL-style setup.
- LoRA does not provide a meaningful gain for Gemma 4 in the current run; the adapted result is effectively flat relative to the base model.
- The largest remaining failure modes appear to be:
  - entity spelling mismatches
  - wrong target column selection
  - prompt continuation or format drift
  - partially correct SQL with one fatal field error

## Comparison With Earlier Repo

Reference results from the previous `mlx-examples/lora` experiment:

- previous run: `Test Loss = 1.674`, `PPL = 5.333`, `EM = 0.0300`, `F1 = 0.3245`
- current baseline: `EM = 0.0000`, `F1 = 0.1089`
- current LoRA run: `Test Loss = 1.676`, `PPL = 5.343`, `EM = 0.0000`, `F1 = 0.3135`
- previous Mistral q4 run: `Test Loss = 1.549`, `PPL = 4.705`, `EM = 0.2400`, `F1 = 0.7231`
- current Mistral q4 run: `Test Loss = 1.561`, `PPL = 4.765`, `EM = 0.1600`, `F1 = 0.4467`
- previous Qwen3 run: `Test Loss = 1.174`, `PPL = 3.234`, `EM = 0.1900`, `F1 = 0.6109`
- current Qwen3 baseline: `EM = 0.0000`, `F1 = 0.1561`
- current Qwen3 run: `Test Loss = 1.217`, `PPL = 3.376`, `EM = 0.1400`, `F1 = 0.6946`
- current Gemma 4 baseline: `EM = 0.0000`, `F1 = 0.1065`
- current Gemma 4 run: `Test Loss = 4.670`, `PPL = 106.661`, `EM = 0.0000`, `F1 = 0.1061`

This is close enough to treat the new repo as a working continuation of the earlier setup, although the current Mistral q4 run is not yet matching the strongest earlier score, while the current Qwen3 run is competitive and even stronger on F1.

## Mistral q4 Notes

- Initial validation loss at iteration `1`: `2.219`
- Final validation loss at iteration `1000`: `1.046`
- Test loss: `1.561`
- Test PPL: `4.765`
- EM: `0.1600`
- F1: `0.4467`
- Final adapter: `outputs/adapters/adapters_mistral_q4.npz`
- Exported predictions: `outputs/predictions/mistral_q4_1000_test_predictions.jsonl`

Example strong prediction from the current run:

```text
SELECT Nationality FROM 1-10015132-16 WHERE Player = 'Terrence Ross'
```

## Qwen3 Notes

- Initial validation loss at iteration `1`: `2.523`
- Final validation loss at iteration `1000`: `1.168`
- Test loss: `1.217`
- Test PPL: `3.376` computed on `20` test batches
- Base EM: `0.0000`
- Base F1: `0.1561`
- QLoRA EM: `0.1400`
- QLoRA F1: `0.6946`
- Final adapter: `outputs/adapters/adapters_qwen3_4b_8bit.npz`
- Exported baseline predictions: `outputs/predictions/qwen3_4b_8bit_base_test_predictions.jsonl`
- Exported predictions: `outputs/predictions/qwen3_4b_8bit_1000_test_predictions.jsonl`

Example strong prediction from the current run:

```text
SELECT Nationality FROM 1-10015132-16 WHERE Player = 'Terrence Ross'
```

## Gemma 4 Notes

- Final validation loss at iteration `1000`: training completed successfully
- Test loss: `4.670`
- Test PPL: `106.661` computed on `10` test batches
- Final adapter: `outputs/adapters/adapters_gemma4_e4b_4bit.npz`
- Cached and non-cached tiny-rollout generation now match after decode-path fixes
- Base full eval on `100` examples: `EM = 0.0000`, `F1 = 0.1065`
- Adapted full eval on `100` examples: `EM = 0.0000`, `F1 = 0.1061`
- Full generative evaluation now runs end-to-end, but the output remains dominated by table-id copying and numeric repetition

Practical status in the current repo:

- model loading works
- forward pass works
- LoRA wrapping works
- training works
- limited perplexity evaluation works
- generation correctness is substantially improved and full EM/F1 evaluation now works
- the main remaining issue is poor SQL output quality rather than a broken generation path

## Comparative Summary

- `TinyLlama` is the weakest of the three but still useful to verify that the LoRA pipeline works in the standalone repo.
- `Mistral q4` currently gives the best exact match score in this repo run.
- `Qwen3` currently gives the best F1 and the best perplexity, which makes it the strongest option for structurally correct SQL generation.
- `Gemma 4` is now supported by the codebase and can be fully evaluated after the decode fix, but its current task performance is far too weak to compete with the other completed runs.
- The combined picture suggests:
  - choose `Mistral q4` if exact match is the main priority in the current runs
  - choose `Qwen3` if near-correct structured output quality is the main priority
  - keep `TinyLlama` as the lightweight baseline and reproducibility reference

## Artifacts

- Local TinyLlama MLX model: `mlx_model`
- Local Mistral q4 MLX model: `mlx_model_mistral_q4`
- Dataset splits:
  - `data/train.jsonl`
  - `data/valid.jsonl`
  - `data/test.jsonl`
- Final adapter: `outputs/adapters/adapters_tinyllama.npz`
- Final Mistral q4 adapter: `outputs/adapters/adapters_mistral_q4.npz`
- Final Qwen3 adapter: `outputs/adapters/adapters_qwen3_4b_8bit.npz`
- Final Gemma 4 adapter: `outputs/adapters/adapters_gemma4_e4b_4bit.npz`
- Exported baseline predictions: `outputs/predictions/tinyllama_base_test_predictions.jsonl`
- Exported predictions: `outputs/predictions/tinyllama_1000_test_predictions.jsonl`
- Exported Mistral q4 baseline predictions: `outputs/predictions/mistral_q4_base_test_predictions.jsonl`
- Exported Mistral q4 predictions: `outputs/predictions/mistral_q4_1000_test_predictions.jsonl`
- Exported Qwen3 baseline predictions: `outputs/predictions/qwen3_4b_8bit_base_test_predictions.jsonl`
- Exported Qwen3 predictions: `outputs/predictions/qwen3_4b_8bit_1000_test_predictions.jsonl`

Notes:

- `*` Qwen3 perplexity metrics were computed on `20` test batches.
- `**` Gemma 4 perplexity metrics were computed on `10` test batches; full EM/F1 on `100` examples are now available and remain very weak.

## Recommended Next Step

Decide whether to:

1. keep `Gemma 4 e4b 4bit` as a completed but weak result in the comparison table, or
2. investigate Gemma-specific prompt or LoRA-target changes before spending more time on another training run
