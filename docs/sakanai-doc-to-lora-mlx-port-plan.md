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

This script trains the hypernetwork through the frozen model's next-token loss. For each synthetic document it generates LoRA weights, patches selected transformer projections, runs the model forward, and backpropagates token cross-entropy into the hypernetwork.

Expected smoke output should show token loss improvement, for example:

```text
initial_loss=4.895902
final_loss=4.356647
improvement=1.12x
```

To run the same path on the MLX model used by the document-specific baseline:

```bash
source ~/.zshrc && scripts/train_doc_to_lora_tinyllama_token_smoke.sh mlx_model 3
```

The local `mlx_model` smoke has been validated with a minimal configuration:

```text
initial_loss=13.364178
final_loss=10.082440
improvement=1.33x
```

If `mlx_model` is not available, convert a Hugging Face model to 4-bit MLX first:

```bash
source ~/.zshrc && PYTHONPATH=src python3 -m lora_mlx.convert \
  --hf-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --mlx-path mlx_model \
  --quantize \
  --q-bits 4
```

The token-level smoke test still uses synthetic documents and synthetic target token ids. It is the first end-to-end gradient check, not a natural document QA result.

## Claim Boundary

Until steps 3-7 are complete, report this as a native MLX D2L port in progress. Do not claim equivalence to SakanaAI results or instant document internalization from the current baseline alone.
