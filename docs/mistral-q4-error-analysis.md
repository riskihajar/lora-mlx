# Mistral q4 1000 Error Analysis

## Scope

- Model: `mlx_model_mistral_q4`
- Adapter: `adapters_mistral_q4_1000.npz`
- Predictions file: `mistral_q4_1000_test_predictions.jsonl`
- Evaluation set size: `100`

## Headline Metrics

- `EM = 0.2400`
- `F1 = 0.7231`
- Exact matches: `24 / 100`

## Error Category Summary

| Category | Count | Notes |
| --- | ---: | --- |
| `exact_match` | 24 | Prediction exactly matches the gold SQL |
| `where_clause_mismatch` | 21 | Selected column is often correct, but filter field or filter value is wrong |
| `multiple_sql_fields_mismatch` | 19 | More than one SQL component is wrong |
| `non_sql_or_incomplete` | 19 | Output falls into prompt continuation, natural text, or incomplete SQL |
| `selected_column_mismatch` | 9 | `SELECT` target is wrong while the rest is mostly correct |
| `aggregation_mismatch` | 8 | `COUNT` is missing, added incorrectly, or replaced with a different pattern |

## Main Patterns

### 1. Where clause mistakes are the biggest structured error

The model often learns the overall SQL skeleton, but confuses which field belongs in the `WHERE` clause.

Examples:

```text
GOLD: SELECT COUNT Nationality FROM 1-1013129-3 WHERE NHL team = 'New Jersey Devils'
PRED: SELECT COUNT Nationality FROM 1-1013129-3 WHERE Player = 'New Jersey Devils'
```

```text
GOLD: SELECT Round FROM 1-10083598-1 WHERE Circuit = 'Assen'
PRED: SELECT Circuit FROM 1-10083598-1 WHERE Report = 'Assen'
```

This suggests the model often understands the entity value but not the exact schema role attached to that value.

### 2. Selected column errors are usually close misses

The model is often near-correct and predicts a semantically related column.

Examples:

```text
GOLD: SELECT School/Club Team FROM 1-10015132-16 WHERE Years in Toronto = '1995-96'
PRED: SELECT Player FROM 1-10015132-16 WHERE Years in Toronto = '1995-96'
```

```text
GOLD: SELECT School/Club Team FROM 1-10015132-16 WHERE Years in Toronto = '2003-06'
PRED: SELECT Club Team FROM 1-10015132-16 WHERE Years in Toronto = '2003-06'
```

These errors often still receive high token-level F1 because the table and condition are correct.

### 3. Aggregation mistakes are relatively rare but expensive

The model sometimes adds `COUNT` when it should not, or forgets it when it should exist.

Examples:

```text
GOLD: SELECT Combined days FROM 1-10182508-5 WHERE Wrestler = 'Go Shiozaki'
PRED: SELECT COUNT Combined days FROM 1-10182508-5 WHERE Wrestler = 'Go Shiozaki'
```

```text
GOLD: SELECT COUNT College/junior/club team FROM 1-1013129-2 WHERE NHL team = 'Washington Capitals'
PRED: SELECT College/junior/club team FROM 1-1013129-2 GROUP BY College/junior/club team
```

This points to partial understanding of question intent, but inconsistent mapping between phrasing and SQL aggregation.

### 4. Some failures are not really SQL failures, but generation-control failures

`19` predictions are prompt continuations, raw text, or incomplete output instead of clean SQL.

Examples:

```text
PRED: 1-1015521-2
Q: What is the rank in the commonwealth's air force if you're a major general in the US air force?
A: SELECT Rank in English FROM 1-1015521-2 WHERE US Air Force
```

```text
PRED: 1
Q: In terms of reigns, what is the highest number listed?
A: 10
Q: In terms of reigns, what is the median number listed?
A: 2
```

This is important because it means some of the remaining gap may be improved by generation setup or output constraint, not only by more training.

### 5. Case-only or tiny lexical differences still hurt EM

Some errors are almost semantically correct but fail exact match due to case or small string differences.

Examples:

```text
GOLD: SELECT COUNT Position FROM 1-1013129-2 WHERE College/junior/club team = 'Sherbrooke Faucons (QMJHL)'
PRED: SELECT COUNT Position FROM 1-1013129-2 WHERE College/junior/club team = 'Sherbrooke Faucons (qmjhl)'
```

```text
GOLD: SELECT Nationality FROM 1-1013129-2 WHERE College/junior/club team = 'Thunder Bay Flyers (USHL)'
PRED: SELECT Nationality FROM 1-1013129-2 WHERE College/junior/club team = 'Thunder Bay Flyers (ushl)'
```

These get `F1 = 1.0` under the current normalization but `EM = 0`.

## What This Means

- The model has learned the SQL format fairly well.
- The remaining errors are increasingly about schema precision and output control.
- The biggest next wins are likely to come from:
  1. reducing prompt continuation / incomplete generations
  2. improving schema grounding for selected columns and `WHERE` fields
  3. tightening aggregation behavior for `COUNT`, `MIN`, and related intents

## Recommended Next Experiments

1. Try a stronger output constraint at inference time, such as stopping generation earlier or post-processing only the first SQL-like line.
2. Increase `lora-layers` from `4` to `8` and compare whether schema-field alignment improves.
3. Add a normalization-based exact match metric for case-insensitive literals in addition to strict EM.
4. Export and inspect only the `non_sql_or_incomplete` examples first, because they are the cleanest target for fast improvement.
