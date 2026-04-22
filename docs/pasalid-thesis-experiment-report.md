# Pasal.id Experiment Report

## Context

- Goal: evaluate whether adapter-based internalization can move document knowledge into an open-weight model so the model can answer without receiving the source document at inference time.
- Data source: Pasal.id API with local derived artifacts only.
- Current corpus stage: verified `UU` subset with cleaned canonical document units and generated QA bank.
- Reading note: `F1` is treated as the main answer-quality metric. `EM` is still reported, but only as a strict auxiliary metric because legal QA answers can vary in wording while remaining substantively correct.
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

### Legal-aware metrics on a `10` example seen subset

To reduce over-reliance on strict EM, the same TinyLlama A/B/C comparison was also read through legal-aware metrics.

| Condition | Answer EM | Answer F1 | Citation EM | Citation Component Score |
| --- | ---: | ---: | ---: | ---: |
| A | 0.0000 | 0.3003 | 0.0000 | 0.0000 |
| B | 0.0000 | 0.4058 | 0.0000 | 0.0000 |
| C | 0.0000 | 0.3577 | 0.0000 | 0.0000 |

Interpretation:

- The legal-aware `answer_f1` result reinforces the same ordering seen in the generic metric run: `B > C > A`.
- `EM` remains uniformly zero, which supports the decision to keep `EM` only as a strict auxiliary metric.
- Citation metrics are still zero because the current generated answers often fail to produce a parsable source reference even when part of the answer content is directionally useful.
- This confirms that answer quality is improving faster than source-traceability quality, so citation formatting and source-attribution behavior still need targeted improvement.

### Refined QA-bank legal-aware check on a `10` example seen subset

After tightening the QA generation prompt so that answers should end with an exact `Sumber:` line, the TinyLlama A/B/C comparison was rerun on a refined split.

| Condition | Answer EM | Answer F1 | Citation EM | Citation Component Score |
| --- | ---: | ---: | ---: | ---: |
| A | 0.0000 | 0.2569 | 0.0000 | 0.0000 |
| B | 0.0000 | 0.3840 | 0.0000 | 0.0000 |
| C | 0.0000 | 0.2663 | 0.0000 | 0.0000 |

Interpretation:

- The refined QA style preserves the same broad ordering: `B > C > A`.
- The adapter condition still improves slightly over the no-context baseline, though the gap is smaller on the refined subset.
- Citation metrics remain zero, which indicates that the main bottleneck is no longer only QA-bank formatting; the model itself is still not reliably reproducing the required citation style at inference time.
- This is an important result for the thesis because it separates two phenomena: partial answer internalization is visible, but source-attribution behavior is still weak.

### Structured answer-format check on a `10` example seen subset

The QA bank was then regenerated with a stricter two-line answer format:

- `Jawaban: ...`
- `Sumber: ...`

This produced a larger structured QA bank and a new structured experiment split.

Structured split summary:

- total rows: `120`
- train rows: `60`
- valid rows: `15`
- test seen rows: `30`
- test unseen rows: `15`

TinyLlama legal-aware metrics on the structured seen subset (`10` examples):

| Condition | Answer EM | Answer F1 | Citation EM | Citation Component Score |
| --- | ---: | ---: | ---: | ---: |
| A | 0.0000 | 0.3354 | 0.0000 | 0.0000 |
| B | 0.0000 | 0.3637 | 0.0000 | 0.0000 |
| C | 0.0000 | 0.3246 | 0.0000 | 0.0000 |

Interpretation:

- The structured answer format keeps answer quality in a usable range.
- The broad ordering still favors the context condition, but on this structured subset the no-context base model is slightly above the adapter condition.
- Most importantly, citation metrics still remain zero, which suggests that even stronger answer formatting alone is not enough to make the model reliably emit parseable source attribution.
- This strengthens the interpretation that the current bottleneck is model adherence to source-format behavior, not only the wording of the gold answers.

### JSON-structured answer-format check on a `10` example seen subset

The answer format was then made even stricter by storing each gold answer as serialized JSON with explicit `answer` and `source` fields.

JSON split summary:

- total rows: `90`
- train rows: `44`
- valid rows: `9`
- test seen rows: `22`
- test unseen rows: `15`

TinyLlama legal-aware metrics on the JSON seen subset (`10` examples):

| Condition | Answer EM | Answer F1 | Citation EM | Citation Component Score |
| --- | ---: | ---: | ---: | ---: |
| A | 0.0000 | 0.3674 | 0.0000 | 0.0000 |
| B | 0.0000 | 0.5026 | 0.0000 | 0.0000 |
| C | 0.0000 | 0.3079 | 0.0000 | 0.0250 |

Interpretation:

- The JSON format preserves the expected ordering `B > A > C` on this subset.
- `B` improves noticeably and remains the strongest condition.
- `C` remains below `A` on this subset, so adapter-only performance is not yet recovering the quality of either the context condition or the stronger earlier TinyLlama seen result.
- The most important movement is that `citation_component_score` becomes non-zero for `C`, which is the first sign that a more constrained answer format may help citation behavior become measurable.
- Even so, citation performance remains very weak overall, so the source-traceability problem is not solved yet.

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

### Longer Mistral adapter check on `20` seen examples

A longer Mistral adapter pass was also evaluated using `outputs/adapters/adapters_pasalid_mistral_q4_experiment_long.npz`.

| Condition | Description | EM | F1 |
| --- | --- | ---: | ---: |
| C-long | Base model + longer LoRA adapter, no context | 0.0000 | 0.2025 |

Interpretation:

- The longer Mistral run improves over the earlier short adapter result (`0.2025` vs `0.1802`).
- The gain is real but still modest.
- Even with the longer pass, Mistral remains behind the TinyLlama adapter baseline on the seen split.

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

## Qwen3 Baseline

### Early status

- Model: `mlx-community/Qwen3-4B-8bit`
- Adapter target: `outputs/adapters/adapters_pasalid_qwen3_experiment.npz`
- A shortened training run was completed to `150` iterations and produced a usable adapter artifact.
- Validation loss at iteration `1`: `1.414`
- Train loss reached the `0.883 - 1.211` range in the short run.

### Seen-split A/B/C smoke results on `10` examples

| Condition | Description | EM | F1 |
| --- | --- | ---: | ---: |
| A | Base model, no context | 0.0000 | 0.1752 |
| B | Base model, with source context | 0.0000 | 0.3684 |
| C | Base model + LoRA adapter, no context | 0.0000 | 0.2411 |

Interpretation:

- `A vs B` again shows a strong context advantage.
- `A vs C` shows a meaningful adapter gain over the no-context base model.
- `B vs C` still leaves a visible gap, but the adapter moves the model toward the context-based condition.
- In this early smoke test, Qwen3 behaves more competitively than Mistral q4 in the adapter-only condition.

### Unseen-split status

- Full unseen smoke checks for Qwen3 did not finish within the current command window.
- The seen-split result is usable, but the unseen comparison for Qwen3 is still incomplete.

Additional failure note:

- Qwen3 still shows answer-format drift, including numbered multiple-choice style output and partial restatement of the prompt structure.

## Current Limitation

- some training samples are still longer than the TinyLlama context window, which may reduce training quality
- the QA bank is generated and useful, but still needs continued quality refinement
- the current TinyLlama run is a first baseline, not the final adapter result for the study
- the current Mistral q4 adapter run is only a short baseline pass and should not be treated as the final Mistral result
- the Qwen3 unseen-split comparison still needs a smaller or more staged evaluation run to complete reliably
- even after tightening the gold-answer style, citation metrics remain zero on the refined TinyLlama legal-aware check
- even after moving to a more structured two-line answer format, citation metrics remain zero in the TinyLlama structured-format check
- the JSON-format experiment shows a small citation gain, but not enough yet to treat source-traceability as solved
- the current Mistral longer run improves on the short pass, but still does not make Mistral the strongest adapter-only baseline

## Next Step

1. refine the QA bank and answer style to reduce variability and improve traceable evidence format
2. complete a staged unseen-split evaluation for Qwen3 so all three conditions are available on both seen and unseen splits
3. only continue extending Mistral if the refined QA format suggests it can close the remaining gap

Refined follow-up direction:

- keep the current finding that answer quality and source traceability are currently separating
- test whether a more explicitly structured output format can improve citation metrics without losing answer quality

Current direction after the JSON-format check:

- the JSON format is the strongest current candidate for future reruns because it is the first format that yields any measurable citation-component movement
- the next useful step is to rerun the JSON-format experiment on a larger subset before treating the result as stable
