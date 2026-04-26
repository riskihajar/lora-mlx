# SakanaAI Doc-to-LoRA MLX Port Plan

This document tracks the target for a native MLX port of SakanaAI-style Doc-to-LoRA (D2L). It is intentionally stricter than the current document-specific LoRA baseline.

## Target Semantics

SakanaAI Doc-to-LoRA internalizes a document by running it through a context encoder and hypernetwork that directly generates or modulates LoRA weights for the base model. A new document should produce adapter weights without optimizing LoRA parameters for that document.

The current baseline in this repo is useful but different: it builds supervised examples from one document and trains a `.npz` LoRA adapter for that document. That remains a baseline, not the final D2L implementation.

## SakanaAI Components Observed

| SakanaAI component | Role | MLX port target |
| --- | --- | --- |
| `ctx_encoder.py` | Encodes context using embedding-only, early-exit, or per-layer activations. | Start with a lightweight MLX text/hash encoder for smoke tests, then replace with MLX model activations. |
| `aggregator.py` | Uses a Perceiver bottleneck to produce outputs per layer, module, and rank. | Add a small MLP/latent aggregator first; replace with Perceiver once generated LoRA training works. |
| `hypernet.py` | Maps aggregated context features into LoRA matrices for target modules. | Generate `A` and `B` matrices for selected MLX transformer projections. |
| `lora_layer.py` | Patches model forward calls with generated LoRA tensors. | Add `GeneratedLoRALinear` and model patch helpers for MLX `nn.Linear` modules. |
| `lora_merger.py` | Combines chunk-level LoRA adapters across document chunks. | Defer until single-context generation works; then add chunk merge/weighted combine. |
| Training objective | Learns hypernetwork parameters from context/question/answer behavior. | Add synthetic key-value smoke training before Pasal.id-scale training. |

## Model Alignment

The upstream SakanaAI Doc-to-LoRA main and NIAH scripts use `google/gemma-2-2b-it` with LoRA on `down_proj` and rank `8`:

```text
--model_name_or_path=google/gemma-2-2b-it
--target_modules=down_proj
--lora_r=8
```

For this MLX port, the matching practical model target is:

```text
mlx-community/gemma-2-2b-it
```

TinyLlama remains useful only as a small local baseline. It should not be treated as the SakanaAI-equivalent target model.

## Dataset Alignment

The upstream SakanaAI main experiment uses the `SakanaAI/self_gen_qa_d2l` dataset. For the Gemma setup, the relevant prefix is:

```text
google/gemma-2-2b-it_temp_0.0_closed_qa_prob_1.0
```

The main config mixes self-generated examples from:

```text
fw_qa_v2/min_0_to_2000/train/*level_1*.parquet
pwc_compact
squad_compact
ropes_compact
drop_compact
```

Use the local bridge script to list and selectively fetch small slices instead of downloading the full upstream dataset, which is about 100 GB per model according to the upstream README:

```bash
source ~/.zshrc && PYTHONPATH=src python3 scripts/prepare_sakana_d2l_dataset.py \
  --list-files \
  --datasets squad_compact \
  --max-files 1
```

Validated file discovery:

```text
google/gemma-2-2b-it_temp_0.0_closed_qa_prob_1.0/squad_compact/train/ds_0000.parquet
```

To download and convert a small sample to JSONL for the MLX token-level objective:

```bash
source ~/.zshrc && PYTHONPATH=src python3 scripts/prepare_sakana_d2l_dataset.py \
  --download \
  --convert \
  --datasets squad_compact \
  --max-files 1 \
  --max-examples 64 \
  --output data/doc_to_lora/sakana_gemma_squad_sample.jsonl
```

Then run the hypernetwork token objective on the Sakana-style sample. The full-answer wrapper trains on all response tokens using original token IDs preserved from Sakana's parquet files:

```bash
source ~/.zshrc && scripts/train_doc_to_lora_sakana_gemma_full_answer.sh \
  data/doc_to_lora/sakana_gemma_squad_sample.jsonl \
  30 \
  1 \
  mlx-community/gemma-2-2b-it
```

Validated Gemma full-answer run on the converted Sakana `squad_compact` sample:

```text
max_examples=1
iters=30
response_tokens=22
initial_loss=12.703471
final_loss=3.094043
improvement=4.11x
initial_token_acc=0.000
final_token_acc=0.091
final_acc=0.000
```

This is now a full-answer teacher-forced objective. Exact full-answer match is still too strict for this tiny overfit run, but the result verifies the full path: upstream SakanaAI dataset tokens -> Gemma MLX -> generated LoRA on `down_proj` -> full response token loss -> hypernetwork update.

For a small multi-example run with held-out eval metrics:

```bash
source ~/.zshrc && scripts/train_doc_to_lora_sakana_gemma_multi_answer.sh \
  data/doc_to_lora/sakana_gemma_squad_sample.jsonl \
  12 \
  5 \
  1 \
  mlx-community/gemma-2-2b-it
```

Validated output with the original deterministic hash context encoder:

```text
train_examples=4
eval_examples=1
iters=12
response_tokens=52
initial_loss=13.131772
final_loss=12.259892
improvement=1.07x
initial_token_acc=0.000
final_token_acc=0.058
initial_eval_loss=13.930460
final_eval_loss=17.204411
eval_improvement=0.81x
```

Validated output after replacing the deterministic hash feature with a trainable hashed-token context encoder:

```text
train_examples=4
eval_examples=1
iters=12
response_tokens=52
initial_loss=13.130325
final_loss=5.914184
improvement=2.22x
initial_token_acc=0.000
final_token_acc=0.077
initial_eval_loss=13.960747
final_eval_loss=7.599777
final_eval_token_acc=0.200
eval_improvement=1.84x
context_encoder=token-hash
```

Validated output after adding an 8-latent query aggregator over trainable token context features:

```text
train_examples=4
eval_examples=1
iters=12
response_tokens=52
initial_loss=13.440420
final_loss=5.293917
improvement=2.54x
initial_token_acc=0.000
final_token_acc=0.077
initial_eval_loss=14.391494
final_eval_loss=5.402127
final_eval_token_acc=0.200
eval_improvement=2.66x
context_encoder=token-hash
context_latents=8
```

SakanaAI's main scripts use `per_rank_gen=True`, so the MLX path now supports `--per-rank-gen`. This generates each LoRA rank from a rank-conditioned latent instead of emitting all rank rows from one head output. Validated output with `per_rank_gen=True`:

```text
train_examples=4
eval_examples=1
iters=12
response_tokens=52
initial_loss=13.541703
final_loss=5.143357
improvement=2.63x
initial_token_acc=0.000
final_token_acc=0.077
initial_eval_loss=14.407256
final_eval_loss=5.265038
final_eval_token_acc=0.200
eval_improvement=2.74x
context_encoder=token-hash
context_latents=8
per_rank_gen=True
```

This is the current default wrapper setting because it matches SakanaAI config and slightly improves held-out loss over the latent-only non-per-rank run (`2.74x` vs `2.66x` eval improvement).

SakanaAI's main scripts also use `per_layer_processing=True`. The MLX path now supports `--per-layer-processing` with a residual MLP block before the LoRA heads. The first run at the previous high smoke LR (`2e-3`) produced NaNs, so the Gemma wrappers now default to `2e-4`, closer to the lower-LR regime used by upstream chunk training. Validated output:

```text
train_examples=4
eval_examples=1
iters=12
learning_rate=2e-4
response_tokens=52
initial_loss=13.237518
final_loss=4.678885
improvement=2.83x
initial_token_acc=0.000
final_token_acc=0.077
initial_eval_loss=14.262685
final_eval_loss=5.230211
final_eval_token_acc=0.200
eval_improvement=2.73x
context_encoder=token-hash
context_latents=8
per_rank_gen=True
per_layer_processing=True
num_pre_head_layers=1
```

This brings the MLX skeleton closer to the upstream hypernetwork settings: `down_proj`, `per_rank_gen=True`, `per_layer_processing=True`, generated LoRA trained against Sakana self-generated Gemma examples.

The MLX path also supports a model-derived context encoder mode with `--context-encoder model-embed`. This uses frozen Gemma input embeddings as document features, making it closer to SakanaAI's `ctx_encoder_type=embed_only` than the trainable token-hash encoder. Validated output:

```text
train_examples=4
eval_examples=1
iters=12
learning_rate=2e-4
response_tokens=52
initial_loss=13.432360
final_loss=4.890327
improvement=2.75x
initial_token_acc=0.000
final_token_acc=0.077
initial_eval_loss=14.356888
final_eval_loss=5.330220
final_eval_token_acc=0.200
eval_improvement=2.69x
context_encoder=model-embed
per_rank_gen=True
per_layer_processing=True
```

This is slightly below the best lightweight token-latent run on eval (`2.69x` vs `2.73x`), but it is a parity improvement because context features now come from the Gemma model rather than a separate toy embedding table.

The MLX path also supports `--context-encoder model-activations`, which runs the document through frozen Gemma layers and uses the matching layer activation feature when generating each target layer's LoRA weights. This is closer to SakanaAI's `ctx_encoder_type=per_layer_activations`, but still not full layer-to-layer processing because the current MLX path uses mean-pooled layer states rather than a full Perceiver over context activations. Validated output from the first averaged-activation implementation:

```text
train_examples=4
eval_examples=1
iters=12
learning_rate=2e-4
response_tokens=52
initial_loss=13.743140
final_loss=4.932809
improvement=2.79x
initial_token_acc=0.000
final_token_acc=0.096
initial_eval_loss=14.470487
final_eval_loss=6.025581
final_eval_token_acc=0.200
eval_improvement=2.40x
context_encoder=model-activations
context_max_tokens=512
per_rank_gen=True
per_layer_processing=True
```

This confirmed the model-activation path works. The implementation has since been updated to preserve per-layer activation features separately before the hypernetwork heads.

With per-layer activation features preserved, the same run is stable at `1e-4` learning rate:

```text
train_examples=4
eval_examples=1
iters=12
learning_rate=1e-4
response_tokens=52
initial_loss=13.768865
final_loss=5.953183
improvement=2.31x
initial_token_acc=0.000
final_token_acc=0.077
initial_eval_loss=14.560974
final_eval_loss=6.237171
final_eval_token_acc=0.200
eval_improvement=2.33x
context_encoder=model-activations
context_max_tokens=512
per_rank_gen=True
per_layer_processing=True
```

At `2e-4`, the per-layer activation matrix path produced `NaN`, so the current safe learning rate for this path is `1e-4` or lower. This is more Sakana-aligned structurally, but it currently underperforms the averaged-activation and model-embed runs on the tiny held-out sample.

Layer/module spec conditioning has also been implemented as an ablation with `--spec-conditioning`. It adds a learned embedding per generated LoRA target before each head. On the same 4-train/1-eval run, it improved train loss slightly but underperformed the latent-only default on eval:

```text
initial_loss=13.184215
final_loss=5.058187
improvement=2.61x
initial_eval_loss=13.607017
final_eval_loss=6.725529
eval_improvement=2.02x
spec_conditioning=True
```

For now the wrappers keep spec conditioning off because the latent-only setting has the better held-out loss (`2.66x` eval improvement).

Interpretation: the native MLX hypernetwork path now trains across multiple Sakana examples and improves a held-out example when the context encoder is learnable. Dataset availability alone was not enough; replacing static hash features with trainable token context features improved both train and eval loss, and adding latent queries improved eval loss further. The next SakanaAI parity gap is replacing this lightweight latent aggregator with model-derived context activations and a fuller Perceiver-style block.

This enables the comparison we need:

1. Sakana dataset plus ordinary per-document LoRA baseline.
2. Sakana dataset plus native MLX hypernetwork-generated LoRA.
3. Pasal.id dataset plus ordinary per-document LoRA baseline.
4. Pasal.id dataset plus native MLX hypernetwork-generated LoRA.

That matrix separates whether gains come from the dataset format, the hypernetwork architecture, or the domain data.

## Minimum Native MLX MVP

1. Infer target linear modules from the loaded MLX model.
2. Encode a document into a fixed-size context feature.
3. Generate per-layer/per-module LoRA `A` and `B` matrices from that feature.
4. Patch the selected model modules so generation uses the generated LoRA deltas.
5. Train the hypernetwork on a tiny synthetic document QA task.
6. Query with no source context and verify that generated adapters change model behavior.

## Parity Gaps

- SakanaAI uses a Transformer/Perceiver context path; the first MLX skeleton uses a simpler encoder so the model shape can be validated locally.
- SakanaAI supports chunk grouping and LoRA merging; this port will add that after single-document generation works.
- SakanaAI integrates PEFT and PyTorch module patching; this repo needs native MLX patching for `q_proj`, `v_proj`, and later other projections.
- True D2L requires training the hypernetwork across many documents. A deterministic or randomly initialized generator is only plumbing, not a result.

## Implementation Order

1. Add MLX generated-LoRA data structures and patch helpers. Done in `src/lora_mlx/doc_to_lora.py`.
2. Add a small context-to-LoRA hypernetwork skeleton. Done in `DocToLoRAHypernetwork`.
3. Add a smoke script with synthetic facts and a tiny train/eval loop. Started in `scripts/train_doc_to_lora_hypernet_smoke.py`.
4. Replace hash/text features with model-derived context activations.
5. Add Perceiver-style aggregation.
6. Add chunk merge support.
7. Run Pasal.id document internalization experiments.

## Synthetic Training Smoke Test

Run a small teacher-adapter imitation task:

```bash
source ~/.zshrc && PYTHONPATH=src python3 scripts/train_doc_to_lora_hypernet_smoke.py \
  --iters 40 \
  --num-docs 4 \
  --hidden-size 64
```

The script creates synthetic documents, assigns each document a deterministic target LoRA adapter, and trains the hypernetwork to reproduce the adapter effect from the document text feature. This verifies that the native MLX path can optimize hypernetwork parameters for `context -> generated LoRA A/B`.

Expected smoke output should show loss reduction, for example:

```text
initial_loss=0.100526
final_loss=0.023121
improvement=4.35x
```

This is still not a result on natural language QA. It only validates the training mechanics needed before adding model-derived context features and a real document QA objective.

## Token-Level End-To-End Smoke Test

Run the smallest token-level version with a random toy MLX transformer:

```bash
source ~/.zshrc && PYTHONPATH=src python3 scripts/train_doc_to_lora_token_smoke.py \
  --toy \
  --iters 20 \
  --num-docs 3 \
  --hidden-size 64 \
  --rank 2 \
  --max-specs 1
```

This script trains the hypernetwork through the frozen model's next-token loss. For each synthetic document it generates LoRA weights, patches selected transformer projections, runs the model forward, and backpropagates token cross-entropy into the hypernetwork. The default target module is `down_proj` to match SakanaAI's Gemma setup.

Expected smoke output should show token loss improvement, for example:

```text
initial_loss=5.219040
final_loss=2.876649
improvement=1.81x
final_acc=0.333
```

To run the same path on the SakanaAI-aligned Gemma MLX model:

```bash
source ~/.zshrc && scripts/train_doc_to_lora_gemma_token_smoke.sh mlx-community/gemma-2-2b-it 3
```

The `mlx-community/gemma-2-2b-it` smoke has been validated with the minimal configuration above:

```text
initial_loss=14.297681
final_loss=0.000137
improvement=104112.53x
final_acc=1.000
```

The older local `mlx_model` smoke has also been validated with a minimal configuration:

```text
initial_loss=12.959651
final_loss=0.000015
improvement=849323.69x
final_acc=1.000
```

If a local Gemma MLX checkpoint is preferred, either download `mlx-community/gemma-2-2b-it` through the loader or convert the original Hugging Face model to MLX. Access to Google's Gemma license may be required for the original `google/gemma-2-2b-it` checkpoint:

```bash
source ~/.zshrc && PYTHONPATH=src python3 -m lora_mlx.convert \
  --hf-path google/gemma-2-2b-it \
  --mlx-path mlx_gemma_2_2b_it_q4 \
  --quantize \
  --q-bits 4
```

The token-level smoke test still uses synthetic documents and synthetic target token ids. It is the first end-to-end gradient check, not a natural document QA result.

## Claim Boundary

Until steps 3-7 are complete, report this as a native MLX D2L port in progress. Do not claim equivalence to SakanaAI results or instant document internalization from the current baseline alone.
