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

Then run the hypernetwork token objective on the Sakana-style sample:

```bash
source ~/.zshrc && PYTHONPATH=src python3 scripts/train_doc_to_lora_token_smoke.py \
  --model mlx-community/gemma-2-2b-it \
  --dataset-jsonl data/doc_to_lora/sakana_gemma_squad_sample.jsonl \
  --max-examples 8 \
  --iters 10 \
  --lora-layers 1 \
  --target-modules down_proj \
  --max-specs 1 \
  --hidden-size 32 \
  --rank 1
```

Validated minimal Gemma run on the converted Sakana `squad_compact` sample:

```text
max_examples=4
iters=3
initial_loss=14.131864
final_loss=4.223494
improvement=3.35x
final_acc=1.000
```

This is still a first-token objective, not full answer generation. Its value is isolating the data path: the same native MLX hypernetwork can now train against upstream SakanaAI self-generated D2L examples.

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
