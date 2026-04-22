# Error Analysis

## Scope

This note summarizes recurring error patterns observed from exported prediction files in `outputs/predictions/` for the current standalone runs.

Models covered:

- `TinyLlama + LoRA 1000`
- `Mistral q4 + QLoRA 1000`
- `Qwen3 4B 8bit + QLoRA 1000`

## TinyLlama

Primary file:

- `outputs/predictions/tinyllama_1000_test_predictions.jsonl`

Recurring failure patterns:

- entity typo with otherwise correct SQL structure
  - example: `Terrance Ross` vs `Terrence Ross`
- prompt continuation and format drift
  - model often restarts the prompt or continues with extra `Q:` / `A:` text
- wrong target column while preserving some table structure
  - example: selecting `Position` instead of `School/Club Team`
- partial SQL correctness with wrong predicate field
  - example: matching on `Player = 'Pick'` or similar malformed conditions

Interpretation:

- TinyLlama clearly learns the target surface format better after LoRA, but still struggles to stay anchored on the exact schema and row condition.
- It is useful as a workflow baseline, but not yet reliable for exact text-to-SQL output.

## Mistral q4

Primary file:

- `outputs/predictions/mistral_q4_1000_test_predictions.jsonl`

Recurring failure patterns:

- strong exact matches on easy/medium cases
  - several examples are fully correct with `EM = 1`
- prompt continuation still appears on some examples
  - especially after short natural-language fragments like years or counts
- predicate-field confusion remains
  - example: using `Round = 'Misano'` instead of `Circuit = 'Misano'`
- over-generalized aggregation or grouping
  - example: using `GROUP BY` instead of a filtered `COUNT`

Interpretation:

- Mistral q4 improves exact SQL generation substantially over TinyLlama.
- The remaining mistakes are often higher-level query-construction mistakes rather than total format failures.
- This makes Mistral q4 a good exact-match-oriented model in the current runs.

## Qwen3 4B 8bit

Primary file:

- `outputs/predictions/qwen3_4b_8bit_1000_test_predictions.jsonl`

Recurring failure patterns:

- many predictions are structurally very close to gold SQL
  - this aligns with the high overall `F1`
- wrong selected column with correct filter clause
  - example: `SELECT Player ... WHERE Years in Toronto = '1995-96'`
- small formatting noise before otherwise correct SQL
  - example: leading `1` before a valid query
- truncation or incomplete quoted string
  - example: unfinished `Thunder Bay Flyers (ushl`
- occasional fallback to direct answer rather than SQL
  - example: raw date answer like `14 September 2012`

Interpretation:

- Qwen3 is the strongest model for near-correct SQL in the current standalone benchmark.
- Many of its misses are "almost correct" and lose EM because of a single wrong selected field, leading token, or truncation.
- This explains why `F1` is strong while `EM` does not surpass Mistral q4 in the current run.

## Cross-Model Comparison

- `TinyLlama` fails most often through prompt drift and coarse schema mistakes.
- `Mistral q4` produces more exact matches, but still has some continuation and predicate-selection issues.
- `Qwen3` most often produces nearly-correct SQL with small fatal errors, which is why it currently leads on `F1`.

## Takeaway

- If the goal is strict exact match in the current experiments, `Mistral q4` is the safer choice.
- If the goal is structurally correct SQL that is often one edit away from correct, `Qwen3` is the strongest current candidate.
- `TinyLlama` remains useful for validating LoRA workflow behavior and observing how much adaptation changes a small model.
