# Technical Notes

## Purpose

This document records the technical workflow, code changes, evaluation flow, and pain points encountered while experimenting with LoRA/QLoRA in `mlx-examples/lora`.

It is intended to help future reproduction and reduce repeated debugging.

## Environment

- Machine: Apple M4
- Memory: 24 GB RAM
- Workspace: `mlx-examples/lora`
- Python: `3.11.14`
- Core packages available during the run:
  - `mlx`
  - `transformers`
  - `numpy`

## High-Level Workflow

The workflow used in this repository was:

1. inspect the example repo and confirm how LoRA is implemented
2. convert or load a compatible MLX model
3. run base generation to observe untuned behavior
4. train LoRA or QLoRA adapters with `lora.py`
5. evaluate with:
   - `test loss`
   - `test ppl`
   - lexical `EM`
   - lexical `F1`
6. export prediction files for error analysis

## Files Created During This Work

- `lora_experiment_report.md`
- `technical_notes.md`
- `evaluate_em_f1.py`
- `export_predictions.py`
- `mistral_q4_1000_error_analysis.md`
- `mistral_q4_1000_test_predictions.jsonl`
- `qwen3_4b_8bit_600_test_predictions.jsonl`
- `qwen3_4b_8bit_1000_test_predictions.jsonl`

## Main Artifacts Produced

### TinyLlama

- Base model directory: `mlx_model`
- Adapters:
  - `adapters_tinyllama_100.npz`
  - `adapters_tinyllama_600.npz`
  - `adapters_tinyllama_1000.npz`

### Mistral q4

- Base model directory: `mlx_model_mistral_q4`
- Adapters:
  - `adapters_mistral_q4_100.npz`
  - `adapters_mistral_q4_600.npz`
  - `adapters_mistral_q4_1000.npz`

### Qwen3

- Base model used for experiments: `mlx-community/Qwen3-4B-8bit`
- Adapters:
  - `adapters_qwen3_4b_8bit_100.npz`
  - `adapters_qwen3_4b_8bit_600.npz`
  - `adapters_qwen3_4b_8bit_1000.npz`

## Commands Used

### Install / sanity check

```bash
source ~/.zshrc && python3 --version
source ~/.zshrc && python3 -c "import mlx, transformers, numpy; print('deps_ok')"
```

### Convert TinyLlama

```bash
source ~/.zshrc && python3 convert.py --hf-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --mlx-path mlx_model
```

### Convert quantized Mistral

```bash
source ~/.zshrc && python3 convert.py --hf-path mistralai/Mistral-7B-v0.1 --mlx-path mlx_model_mistral_q4 -q
```

### Train TinyLlama LoRA

```bash
source ~/.zshrc && python3 lora.py --model mlx_model --train --data data --iters 100 --batch-size 1 --lora-layers 4 --steps-per-report 10 --steps-per-eval 25 --save-every 50 --adapter-file adapters_tinyllama_100.npz
```

Then resumed similarly for `600` and `1000` total iterations.

### Train Mistral q4 QLoRA

```bash
source ~/.zshrc && python3 lora.py --model mlx_model_mistral_q4 --train --data data --iters 100 --batch-size 1 --lora-layers 4 --steps-per-report 10 --steps-per-eval 25 --save-every 50 --adapter-file adapters_mistral_q4_100.npz
```

Then resumed similarly for `600` and `1000` total iterations.

### Train Qwen3 QLoRA

```bash
source ~/.zshrc && python3 lora.py --model mlx-community/Qwen3-4B-8bit --train --data data --iters 100 --batch-size 1 --lora-layers 4 --steps-per-report 10 --steps-per-eval 25 --save-every 50 --adapter-file adapters_qwen3_4b_8bit_100.npz
```

Then resumed similarly for `600` and `1000` total iterations.

### Evaluate perplexity

```bash
source ~/.zshrc && python3 lora.py --model <model> --adapter-file <adapter> --test --batch-size 1 --test-batches <n>
```

### Evaluate EM/F1

```bash
source ~/.zshrc && python3 evaluate_em_f1.py --model <model>
source ~/.zshrc && python3 evaluate_em_f1.py --model <model> --adapter-file <adapter>
```

### Export predictions

```bash
source ~/.zshrc && python3 export_predictions.py --model <model> --adapter-file <adapter> --output <file>.jsonl
```

## Evaluation Scripts Added

### `evaluate_em_f1.py`

This script computes:

- `EM`: strict exact string match after whitespace normalization
- `F1`: token-level F1 after lowercase and light SQL token normalization

Notes:

- this is lexical evaluation only
- this is not execution accuracy
- for Qwen3, `max-new-tokens=32` was used during stable evaluation runs to reduce runtime and generation drift

### `export_predictions.py`

This script exports per-example records as JSONL with:

- `index`
- `prompt`
- `gold`
- `prediction`
- `em`
- `f1`

This was used for later error analysis.

## Qwen3 Compatibility Patches

The original minimal model implementation in `models.py` was built for Llama/Mistral-style models and did not load Qwen3 correctly.

The following compatibility changes were added:

### 1. Support explicit `head_dim`

Qwen3 config includes `head_dim` and does not assume:

```text
head_dim = hidden_size // num_attention_heads
```

This was required to fix projection shape mismatches such as `q_proj.weight`.

### 2. Support `q_norm` and `k_norm`

Qwen3 attention weights include:

- `self_attn.q_norm.weight`
- `self_attn.k_norm.weight`

These were missing from the original model implementation and caused weight loading failure.

### 3. Support tied embeddings

Qwen3 config uses tied word embeddings.

The original implementation always created `lm_head`. That had to be adjusted so that when `tie_word_embeddings=True`, output logits are produced from the embedding matrix instead.

### 4. Handle `QuantizedEmbedding` output projection

When using `mlx-community/Qwen3-4B-8bit`, the embedding layer is a `QuantizedEmbedding`.

That means the tied output projection cannot directly use `embed_tokens.weight.T` as if it were dense FP weights. The code had to dequantize the embedding weights first before computing logits.

## Pain Points and Important Caveats

### 1. This repo is not a general model loader

The example code is narrow by design.

It assumes a Llama/Mistral-like structure and directly reaches into internals such as:

- `model.model.layers[...]`
- `self_attn.q_proj`
- `self_attn.v_proj`

Any newer architecture can fail even if it is conceptually similar.

### 2. Qwen3 did not work out of the box

Both of these initially failed:

- converting from `Qwen/Qwen3-4B-Instruct-2507`
- loading `mlx-community/Qwen3-4B-8bit`

The blocker was architecture mismatch, not just memory or conversion.

### 3. `lora.py` assumes an adapter file exists for prompt/test mode

Even when you only want generation, `lora.py` still raises if `--adapter-file` does not exist.

This makes base-model-only prompt evaluation awkward through the script. Direct calls through `utils.load()` are more convenient for pure base generation.

### 4. Missing dataset files are not handled gracefully

`Dataset` may set `_data = None` if a file is missing, and later `len(dataset)` can fail in a less friendly way.

### 5. Long examples are only warned, not truncated

The training loop prints a warning for sequences above `2048` tokens but does not truncate automatically.

### 6. PPL runs can be slow enough to hit timeouts

For Qwen3, some full-size evaluation runs took too long for the current command timeout. Because of that:

- some Qwen3 `PPL` values were computed on `20` test batches rather than all `100`

This should be remembered when comparing PPL values across models.

### 7. EM and F1 can disagree strongly

This happened repeatedly.

Examples:

- case-only mismatch in literals
- almost correct SQL with one wrong field
- column alias/partial field overlap

So `EM` is very strict, while `F1` is often more informative about structural learning.

### 8. More training is not always better on every metric

For Qwen3:

- `600` steps produced better `F1` than `1000`
- `1000` steps still improved `EM` and `PPL`

This suggests checkpoint selection should depend on the target metric, not just iteration count.

## Practical Lessons Learned

### TinyLlama

- good for understanding the workflow
- weak as a final task model

### Mistral q4

- strongest exact-match performer so far
- best balanced result overall in this repo

### Qwen3 4B 8bit

- needed compatibility patching first
- after patching, became surprisingly competitive
- likely offers a very good quality/speed trade-off

## Reproduction Advice

If you want to reproduce the strongest results with the least friction:

1. start with `mistralai/Mistral-7B-v0.1` quantized
2. use `--batch-size 1` and `--lora-layers 4`
3. train to `600` iterations first
4. evaluate using both `PPL` and `EM/F1`

If you want to reproduce the Qwen3 path:

1. keep the patched `models.py`
2. use `mlx-community/Qwen3-4B-8bit`
3. do not assume HF-to-MLX convert is necessary for the first experiment
4. prefer checking `600` and `1000` checkpoints separately

## Current Recommendation

- Use `Mistral q4` when exact text-to-SQL quality is the priority.
- Use `Qwen3 4B 8bit` when exploring a compact but strong modern alternative.
- Keep all comparisons explicit about:
  - number of test batches used for `PPL`
  - `max-new-tokens` used for `EM/F1`
  - whether the model came from local conversion or MLX community weights
