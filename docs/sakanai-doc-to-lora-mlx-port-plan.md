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

The activation path now also supports a trainable latent attention pooling mode with `--activation-pooling latent`. In this mode, preprocessing preserves sequence activations as `(layers, tokens, hidden)` and the hypernetwork pools each layer's context tokens through learned latent queries before generating LoRA weights. This is closer to the Perceiver-style aggregation used by SakanaAI than mean pooling, but still a compact MLX approximation. Validated output:

```text
train_examples=4
eval_examples=1
iters=12
learning_rate=1e-4
response_tokens=52
initial_loss=13.055923
final_loss=5.659515
improvement=2.31x
initial_token_acc=0.000
final_token_acc=0.077
initial_eval_loss=13.307540
final_eval_loss=6.629034
final_eval_token_acc=0.200
eval_improvement=2.01x
context_encoder=model-activations
context_max_tokens=128
activation_pooling=latent
activation_latents=4
per_rank_gen=True
per_layer_processing=True
```

The latent activation aggregator is structurally closer to SakanaAI, but on the current tiny sample it underperforms simpler mean-pooled activation features. It should be treated as a parity building block, not the current best small-sample setting.

Chunked context support has been added with `--context-chunk-tokens`. For model-derived context modes, the script now extracts one context feature per chunk, generates one LoRA group per chunk, and merges the generated LoRAs by averaging matching A/B matrices. This is a simple version of SakanaAI's chunked context plus LoRA merger path, not a learned merger yet. Validated `model-embed` output:

```text
train_examples=4
eval_examples=1
iters=12
learning_rate=2e-4
response_tokens=52
initial_loss=13.441004
final_loss=4.768547
improvement=2.82x
initial_token_acc=0.000
final_token_acc=0.077
initial_eval_loss=14.375244
final_eval_loss=5.312313
final_eval_token_acc=0.200
eval_improvement=2.71x
context_encoder=model-embed
context_max_tokens=512
context_chunk_tokens=128
per_rank_gen=True
per_layer_processing=True
```

Validated chunked `model-activations` output:

```text
train_examples=4
eval_examples=1
iters=12
learning_rate=1e-4
response_tokens=52
initial_loss=13.653996
final_loss=5.856358
improvement=2.33x
initial_token_acc=0.000
final_token_acc=0.058
initial_eval_loss=14.300089
final_eval_loss=6.213915
final_eval_token_acc=0.200
eval_improvement=2.30x
context_encoder=model-activations
context_max_tokens=512
context_chunk_tokens=128
activation_pooling=mean
per_rank_gen=True
per_layer_processing=True
```

The chunked embedding path slightly improved the tiny held-out result compared with non-chunked model embeddings (`2.71x` vs `2.69x`). The chunked activation path remained structurally useful but below simpler embedding features on this small run.

Chunked context now also supports `--chunk-merge learned`, which adds trainable per-target chunk weights inside the hypernetwork and merges matching generated LoRA A/B matrices with a learned softmax over chunks. This is a closer structural match to SakanaAI's learned LoRA merger than the earlier mean merge. On the 4-train/1-eval smoke run, learned merge matches mean merge because the initial uniform weighting is already strong and the sample is tiny:

```text
train_examples=4
eval_examples=1
iters=12
learning_rate=2e-4
response_tokens=52
initial_loss=13.441004
final_loss=4.767913
improvement=2.82x
initial_token_acc=0.000
final_token_acc=0.077
initial_eval_loss=14.375244
final_eval_loss=5.312351
final_eval_token_acc=0.200
eval_improvement=2.71x
context_encoder=model-embed
context_max_tokens=512
context_chunk_tokens=128
chunk_merge=learned
per_rank_gen=True
per_layer_processing=True
```

A larger local run over all 16 converted Sakana examples produced strong train improvement but weaker held-out improvement, showing that the 4/1 split was optimistic:

```text
train_examples=12
eval_examples=4
iters=20
learning_rate=2e-4
response_tokens=95
initial_loss=13.630783
final_loss=4.887667
improvement=2.79x
initial_token_acc=0.000
final_token_acc=0.126
initial_eval_loss=13.543205
final_eval_loss=8.429949
final_eval_token_acc=0.085
eval_improvement=1.61x
context_encoder=model-embed
context_max_tokens=512
context_chunk_tokens=128
chunk_merge=learned
per_rank_gen=True
per_layer_processing=True
```

Mean merge on the same 12/4 split was effectively identical (`final_eval_loss=8.424996`, `eval_improvement=1.61x`), so learned merger is currently a parity feature rather than a measured small-sample gain. The important progress is that chunk-level LoRA merging is now trainable and differentiable.

The dataset bridge has also been scaled from a single `squad_compact` parquet to a 64-example local multi-dataset sample. For the current Gemma prefix, the available selected parquets were `squad_compact`, `drop_compact`, and `ropes_compact`; `pwc_compact` was requested but not selected for this prefix. The 64-example JSONL was generated with:

```bash
PYTHONPATH=src python3 scripts/prepare_sakana_d2l_dataset.py \
  --datasets squad_compact,pwc_compact,drop_compact,ropes_compact \
  --max-files 1 \
  --max-examples 64 \
  --download \
  --convert \
  --output data/doc_to_lora/sakana_gemma_multi_64.jsonl
```

The first scaled benchmark used 32 examples from that file with a 24/8 train/eval split, chunked model embeddings, learned chunk merger, per-rank generation, and per-layer hypernetwork processing:

```text
train_examples=24
eval_examples=8
iters=16
learning_rate=1e-4
response_tokens=204
initial_loss=13.601946
final_loss=6.512096
improvement=2.09x
initial_token_acc=0.000
final_token_acc=0.118
initial_eval_loss=13.608601
final_eval_loss=8.529862
final_eval_token_acc=0.108
eval_improvement=1.60x
loss_type=ce
context_encoder=model-embed
context_max_tokens=512
context_chunk_tokens=128
chunk_merge=learned
per_rank_gen=True
per_layer_processing=True
```

The matching KL/top-k run was nearly identical:

```text
train_examples=24
eval_examples=8
iters=16
learning_rate=1e-4
response_tokens=204
initial_loss=13.631938
final_loss=6.560293
improvement=2.08x
initial_token_acc=0.000
final_token_acc=0.118
initial_eval_loss=13.591545
final_eval_loss=8.563131
final_eval_token_acc=0.108
eval_improvement=1.59x
loss_type=kl-topk
context_encoder=model-embed
context_max_tokens=512
context_chunk_tokens=128
chunk_merge=learned
per_rank_gen=True
per_layer_processing=True
```

The scaled run confirms the MLX Doc-to-LoRA pipeline now handles multi-dataset Sakana samples and larger eval splits. It also shows the remaining bottleneck is generalization: train loss improves consistently, but held-out improvement is much smaller on 8 eval examples than on the earlier 1-example smoke split.

The token smoke trainer now supports hypernetwork checkpoints with `--save-hypernet` and `--load-hypernet`. Checkpoints save the trainable hypernetwork weights as MLX `.npz` files, including learned merger weights when enabled. Optimizer state is not persisted yet, so loaded runs resume from the saved hypernetwork parameters with a fresh optimizer. Validated toy checkpoint restore reproduced the saved loss exactly:

```text
saved_hypernet=/tmp/lora_mlx_hypernet_toy.npz
loaded_hypernet=/tmp/lora_mlx_hypernet_toy.npz
initial_loss=2.618373
final_loss=2.618373
initial_token_acc=0.500
final_token_acc=0.500
```

Validated Gemma checkpoint restore with chunked model embeddings and learned chunk merge also reproduced the saved metrics:

```text
saved_hypernet=/tmp/lora_mlx_hypernet_gemma_smoke.npz
loaded_hypernet=/tmp/lora_mlx_hypernet_gemma_smoke.npz
initial_loss=12.685263
final_loss=12.685263
initial_eval_loss=14.242848
final_eval_loss=14.242848
context_encoder=model-embed
context_chunk_tokens=64
chunk_merge=learned
per_rank_gen=True
per_layer_processing=True
```

This enables longer multi-dataset runs to be saved and inspected, even before full optimizer-state resume is implemented.

The first checkpointed 64-example multi-dataset run used 48 train and 16 eval examples with chunked model embeddings, learned chunk merger, per-rank generation, and per-layer hypernetwork processing:

```text
train_examples=48
eval_examples=16
iters=12
learning_rate=1e-4
response_tokens=468
initial_loss=13.624434
final_loss=8.137347
improvement=1.67x
initial_token_acc=0.000
final_token_acc=0.111
initial_eval_loss=13.544464
final_eval_loss=9.170728
final_eval_token_acc=0.079
eval_improvement=1.48x
loss_type=ce
context_encoder=model-embed
context_max_tokens=512
context_chunk_tokens=128
chunk_merge=learned
per_rank_gen=True
per_layer_processing=True
saved_hypernet=outputs/doc_to_lora/hypernet_gemma_multi64_learned_ce.npz
```

Reloading the saved checkpoint with `--iters 0` reproduced the saved train/eval metrics exactly:

```text
loaded_hypernet=outputs/doc_to_lora/hypernet_gemma_multi64_learned_ce.npz
initial_loss=8.137347
final_loss=8.137347
initial_token_acc=0.111
final_token_acc=0.111
initial_eval_loss=9.170728
final_eval_loss=9.170728
initial_eval_token_acc=0.079
final_eval_token_acc=0.079
train_examples=48
eval_examples=16
```

This is the largest validated MLX Doc-to-LoRA run so far. The checkpoint path works, but the learning curve was noisy at `1e-4` and held-out improvement fell to `1.48x`, so the next scale step should reduce LR and/or increase iterations rather than only increasing data size.

Repeating the same 64-example checkpointed run with a lower learning rate and more iterations improved stability and held-out loss:

```text
train_examples=48
eval_examples=16
iters=24
learning_rate=5e-5
response_tokens=468
initial_loss=13.624434
final_loss=7.183047
improvement=1.90x
initial_token_acc=0.000
final_token_acc=0.111
initial_eval_loss=13.544464
final_eval_loss=8.673681
final_eval_token_acc=0.079
eval_improvement=1.56x
loss_type=ce
context_encoder=model-embed
context_max_tokens=512
context_chunk_tokens=128
chunk_merge=learned
per_rank_gen=True
per_layer_processing=True
saved_hypernet=outputs/doc_to_lora/hypernet_gemma_multi64_learned_ce_lr5e5.npz
```

Reloading `outputs/doc_to_lora/hypernet_gemma_multi64_learned_ce_lr5e5.npz` with `--iters 0` reproduced the saved metrics exactly (`initial_loss=7.183047`, `initial_eval_loss=8.673681`). Compared with the `1e-4` run, `5e-5` improved train loss (`8.137347 -> 7.183047`) and eval loss (`9.170728 -> 8.673681`) while keeping eval token accuracy unchanged (`0.079`). The next stable scale setting should start from `5e-5`, not `1e-4`.

The stable 64-example benchmark can now be reproduced with a wrapper:

```bash
scripts/train_doc_to_lora_sakana_gemma_scale64.sh train
scripts/train_doc_to_lora_sakana_gemma_scale64.sh eval
```

The wrapper defaults to `data/doc_to_lora/sakana_gemma_multi_64.jsonl`, saves to `outputs/doc_to_lora/hypernet_gemma_multi64_learned_ce_lr5e5.npz`, and uses the current best stable scale setting: chunked model embeddings, learned chunk merger, per-rank generation, per-layer hypernetwork processing, CE loss, `5e-5` learning rate, 48 train / 16 eval examples.

An initial internalization/query API is now available:

```bash
PYTHONPATH=src python3 scripts/internalize_doc_to_lora.py \
  --model mlx-community/gemma-2-2b-it \
  --hypernet outputs/doc_to_lora/hypernet_gemma_multi64_learned_ce_lr5e5.npz \
  --document-file path/to/document.txt \
  --output outputs/doc_to_lora/internalized_doc.npz

PYTHONPATH=src python3 scripts/query_internalized_lora.py \
  --model mlx-community/gemma-2-2b-it \
  --lora outputs/doc_to_lora/internalized_doc.npz \
  --prompt "user\nQuestion here\nmodel\n" \
  --max-tokens 64
```

The first script loads the trained hypernetwork checkpoint, embeds/chunks the document with frozen Gemma embeddings, generates LoRA A/B matrices, and saves the generated adapter plus JSON metadata. The second script reloads only the generated LoRA and patches it into Gemma for no-source-context querying. A plumbing validation produced `outputs/doc_to_lora/internalized_blue_falcon.npz` and successfully loaded it for generation. The generated text quality on that synthetic document was poor, so this should be treated as an API milestone rather than evidence of reliable no-context answer generation.

For objective no-source-context evaluation, `scripts/eval_doc_to_lora_internalization.py` internalizes each JSONL document on the fly, patches the generated LoRA into Gemma, and reports teacher-forced prompt/response metrics without including the source document in the prompt. On the full 64-example local multi-dataset sample:

```text
examples=64
response_tokens=670
internalized_loss=7.555706
internalized_token_acc=0.101
internalized_exact_acc=0.000
hypernet=outputs/doc_to_lora/hypernet_gemma_multi64_learned_ce_lr5e5.npz
context_chunk_tokens=128
chunk_merge=learned
```

On the held-out 16-example split matching the checkpointed training run:

```text
examples=16
skip_examples=48
response_tokens=202
internalized_loss=8.673681
internalized_token_acc=0.079
internalized_exact_acc=0.000
hypernet=outputs/doc_to_lora/hypernet_gemma_multi64_learned_ce_lr5e5.npz
context_chunk_tokens=128
chunk_merge=learned
```

This evaluator reproduces the checkpoint reload metrics exactly and gives a cleaner end-to-end Doc-to-LoRA metric: document internalization happens before evaluation, while the answer prompt itself contains no source context.

The Sakana dataset bridge now preserves teacher `logprobs_vals` and `logprobs_indices` from the parquet rows. The token smoke trainer supports `--loss-type kl-topk`, which trains against the sparse top-k teacher distribution instead of only the hard response token. This is closer to SakanaAI's `use_kl_loss=True` objective, although the current MLX version normalizes only over the provided top-k logits. Validated chunked `model-embed` output:

```text
train_examples=4
eval_examples=1
iters=12
learning_rate=2e-4
response_tokens=52
initial_loss=13.452770
final_loss=5.596339
improvement=2.40x
initial_token_acc=0.000
final_token_acc=0.058
initial_eval_loss=14.169008
final_eval_loss=5.952075
final_eval_token_acc=0.000
eval_improvement=2.38x
loss_type=kl-topk
context_encoder=model-embed
context_max_tokens=512
context_chunk_tokens=128
per_rank_gen=True
per_layer_processing=True
```

The KL/top-k path works and is structurally closer to SakanaAI, but on this tiny sample it underperforms the hard-token CE run (`2.38x` eval KL vs `2.71x` eval CE). It should be treated as a parity path for larger data, not the current best smoke-test setting.

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
