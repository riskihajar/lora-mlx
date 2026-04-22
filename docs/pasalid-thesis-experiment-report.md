# Pasal.id Experiment Report

## Context

- Goal: evaluate whether adapter-based internalization can move document knowledge into an open-weight model so the model can answer without receiving the source document at inference time.
- Data source: Pasal.id API with local derived artifacts only.
- Current corpus stage: verified `UU` subset with cleaned canonical document units and generated QA bank.
- Main experimental conditions:
  - `A`: base model without document context
  - `B`: base model with document context
  - `C`: base model with LoRA adapter, without document context

## Current Data Pipeline

- Raw law cache: `data/pasalid_raw/`
- Canonical document units: `data/pasalid/doc_units.jsonl`
- Generated QA bank: `data/pasalid/qa_bank_full.jsonl`
- Experiment split directory: `data/pasalid/experiment_split/`

Current split manifest:

- total QA rows: `285`
- train rows: `156`
- valid rows: `21`
- test seen rows: `78`
- test unseen rows: `30`

## TinyLlama Baseline

### Training status

- Model: `mlx_model`
- Adapter: `outputs/adapters/adapters_pasalid_tinyllama_experiment.npz`
- Training run completed to `300` iterations for the first baseline pass
- Validation loss improved from `1.397` at iteration `1` to `1.046` at iteration `200`

### Seen-split A/B/C smoke results

The first larger smoke comparison was run on `20` seen-split examples.

| Condition | Description | EM | F1 |
| --- | --- | ---: | ---: |
| A | Base model, no context | 0.0000 | 0.2596 |
| B | Base model, with source context | 0.0000 | 0.3592 |
| C | Base model + LoRA adapter, no context | 0.0000 | 0.3033 |

## Initial Interpretation

- `A vs B`: explicit document context clearly improves answer quality on the seen split.
- `A vs C`: the adapter condition improves over the no-context baseline, which is the first sign that some document knowledge is being internalized.
- `B vs C`: the adapter condition does not yet match the context-based condition, but it narrows the gap meaningfully in this early run.

This is a useful early signal for the thesis direction because the adapter-only setting appears better than the plain no-context setting, even before any larger or stronger-model run.

### Seen-split A/B/C smoke results on `20` examples

The same comparison was then extended to `20` examples from the seen split.

| Condition | Description | EM | F1 |
| --- | --- | ---: | ---: |
| A | Base model, no context | 0.0000 | 0.2596 |
| B | Base model, with source context | 0.0000 | 0.3592 |
| C | Base model + LoRA adapter, no context | 0.0000 | 0.3033 |

Interpretation:

- `B` remains the strongest of the three TinyLlama conditions.
- `C` continues to improve over `A`, which supports the internalization direction.
- the adapter-only condition does not yet match the context condition, but it closes part of the gap.

## Mistral q4 Baseline

### Early status

- Model: `mlx_model_mistral_q4`
- Adapter target: `outputs/adapters/adapters_pasalid_mistral_q4_experiment.npz`
- A shortened training run was completed to `150` iterations and produced a usable adapter artifact.
- Validation loss at iteration `1`: `1.189`
- Train loss reached the `0.593 - 0.881` range in the completed short run.

### Seen-split A/B smoke results on `20` examples

| Condition | Description | EM | F1 |
| --- | --- | ---: | ---: |
| A | Base model, no context | 0.0000 | 0.1032 |
| B | Base model, with source context | 0.0000 | 0.3137 |
| C | Base model + LoRA adapter, no context | 0.0000 | 0.1802 |

Initial interpretation:

- `A vs B` again shows a clear benefit from providing source context at inference time.
- `A vs C` shows that the Mistral adapter improves over the no-context baseline, but the gain is smaller than the TinyLlama gain in the current run.
- `B vs C` shows that the Mistral adapter still remains far behind the context-based condition.
- In this early Pasal.id setup, base Mistral q4 without context is substantially weaker than TinyLlama on the seen-split smoke test, and the partial adapter run does not close the gap enough yet.

### Unseen-split smoke results on `10` examples

#### TinyLlama

| Condition | Description | EM | F1 |
| --- | --- | ---: | ---: |
| A | Base model, no context | 0.0000 | 0.2599 |
| B | Base model, with source context | 0.0000 | 0.2917 |
| C | Base model + LoRA adapter, no context | 0.0000 | 0.2412 |

Interpretation:

- On the unseen split, the context benefit for TinyLlama is still present but smaller than on the seen split.
- The adapter condition does not outperform the no-context baseline on this unseen subset.
- This suggests the current TinyLlama adapter is showing stronger seen-document internalization than unseen-document generalization.

#### Mistral q4

| Condition | Description | EM | F1 |
| --- | --- | ---: | ---: |
| A | Base model, no context | 0.0000 | 0.0885 |
| B | Base model, with source context | 0.0000 | 0.2684 |
| C | Base model + LoRA adapter, no context | 0.0000 | 0.1754 |

Interpretation:

- Mistral again benefits strongly from context on the unseen split.
- The adapter improves over the no-context base, but still lags far behind the context condition.
- Like TinyLlama, the current Mistral run does not yet show strong unseen-document internalization.

## Observed Failure Patterns

- the base model still tends to answer with noisy, generic, or partially irrelevant continuations
- the context-based condition often captures the right article or legal consequence, but phrasing is still messy
- the adapter condition shows improvement, but still hallucinates unrelated references and awkward wording
- exact match remains `0`, which suggests the current answer style is still too variable and insufficiently controlled
- Mistral q4 without an adapter currently over-enumerates or drifts into irrelevant legal references on some Pasal.id questions
- unseen questions about numeric legal or budget values remain difficult for both TinyLlama and Mistral without explicit context

## Current Limitation

- some training samples are still longer than the TinyLlama context window, which may reduce training quality
- the QA bank is generated and useful, but still needs continued quality refinement
- the current TinyLlama run is a first baseline, not the final adapter result for the study
- the current Mistral q4 adapter run is only a short baseline pass and should not be treated as the final Mistral result

## Next Step

1. complete a longer `Mistral q4` adapter run on the same experiment split
2. refine the QA bank and answer style to reduce variability and improve traceable evidence format
3. move to `Qwen3` after the stronger Mistral baseline is stabilized
