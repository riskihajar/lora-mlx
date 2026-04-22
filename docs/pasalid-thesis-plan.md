# Pasal.id Thesis Implementation Plan

## Goal

Build a thesis-oriented workflow to test whether LoRA or QLoRA adapters can internalize core information from long Indonesian legal documents, so that an open-weight model can answer legal questions without receiving the source document at inference time.

This plan treats Pasal.id as a practical integration source for data access and prototyping, while keeping legal and methodological boundaries explicit.

## Experimental Framing

This thesis is framed as an experimental comparison, not a hypothesis-testing study.

### Main experimental objective

- Examine whether adapter-based internalization can make an open-weight model answer document-specific legal questions without receiving the source document at inference time.

### Target capability

- During adapter training, the model may learn from document-grounded examples.
- During final evaluation, the adapter-based model must answer without receiving the original document in the prompt.

### Core experiment design

The study compares three main conditions:

#### A. Base model without document context

- The model receives the question only.
- This is the no-context baseline.

#### B. Base model with source document context

- The model receives the question together with the relevant source document or source excerpt.
- This represents the context-based baseline.

#### C. Base model plus LoRA adapter trained to internalize document content

- The model receives the question only.
- The source document is not included at inference time.
- This is the proposed approach.

### Core comparisons

- `A vs B`
  - shows the benefit of providing source documents at inference time
- `B vs C`
  - evaluates whether adapter-based internalization can approach the context-based method
- `A vs C`
  - shows the effect of the proposed method compared with the no-context condition

### Reading dimensions for the experiment

The results should be interpreted through the following dimensions:

- quality of answer
- factual consistency
- evidence support
- source traceability
- inference efficiency

## Important constraints

### Legal and platform constraints

- Pasal.id Terms of Service state that regulation text is public domain, but the structured database and its organization are Pasal.id's work product.
- The service also prohibits scraping beyond rate limits and redistributing the structured database as a whole.
- Because of that, this implementation should avoid treating Pasal.id as a bulk mirrored corpus that is redistributed directly.

### Recommended safe use of Pasal.id

- Use the Pasal.id API for controlled ingestion, prototyping, and experiment support.
- Store only the minimum derived data needed for the thesis workflow.
- Do not publish or redistribute a raw dump of the structured Pasal.id database.
- If the thesis later requires larger-scale publication of derived data, request explicit permission first.

### Operational constraint

- The Pasal.id API requires a bearer token.
- The API key is already available locally in `.env`.
- The implementation must load the key from environment variables and must not hardcode it or commit it.

## Proposed task design

The thesis target is not generic legal search. The primary task should test parametric internalization.

### Primary task

- Document-grounded legal QA with document-free inference

Training pattern:

- model sees legal document content plus question plus answer

Inference pattern:

- model sees question only
- source document is not included

### Secondary tasks

- citation prediction
- article retrieval prediction
- article-level summarization or extraction

These secondary tasks can support analysis, but they should not replace the primary thesis task.

## Dataset strategy

### Recommended dataset levels

Build the dataset in three levels so the thesis can measure both memorization and abstraction.

#### Level A: atomic facts

- definitions
- obligations
- prohibitions
- exceptions
- sanctions

Example:

```text
Pertanyaan: Apa definisi data pribadi?
Jawaban: Data pribadi adalah data tentang orang perseorangan yang teridentifikasi atau dapat diidentifikasi.
```

#### Level B: article-level QA

- one question answerable from a single article or a short local span

Example:

```text
Pertanyaan: Apa kewajiban utama pengendali data pribadi menurut pasal ini?
Jawaban: Pengendali data pribadi wajib ...
```

#### Level C: cross-article QA

- one question requiring information from multiple sections or articles in the same regulation

Example:

```text
Pertanyaan: Dalam kondisi apa data pribadi dapat diproses, dan apa kewajiban lanjutan pengendalinya?
Jawaban: Data pribadi dapat diproses ketika ..., dan pengendalinya wajib ...
```

This level is important because it is stronger evidence of internalized document knowledge than simple local quoting.

## Data acquisition plan

### Pasal.id endpoints to use

- `GET /api/v1/laws`
  - to enumerate candidate regulations
- `GET /api/v1/laws/{frbr_uri}`
  - to retrieve full structured content for a selected regulation
- `GET /api/v1/search`
  - optional, for exploration or retrieval baselines

### Initial scope recommendation

Start with a narrow and high-value legal subset, for example:

- `UU` only
- focus topics such as:
  - pelindungan data pribadi
  - ketenagakerjaan
  - perlindungan konsumen
  - hukum pidana modern

This reduces annotation burden and makes thesis evaluation easier to control.

### Ingestion workflow

1. Load API token from `.env`
2. Enumerate regulations using `/api/v1/laws`
3. Filter by:
   - `type`
   - `year`
   - `status`
4. Fetch selected regulation detail using `/api/v1/laws/{frbr_uri}`
5. Save only derived experiment artifacts locally

### Suggested raw cache format

Create a local, non-public cache directory such as:

```text
data/pasalid_raw/
```

Example files:

```text
data/pasalid_raw/laws_index.json
data/pasalid_raw/akn_id_act_uu_2022_27.json
data/pasalid_raw/akn_id_act_uu_2003_13.json
```

This cache is for reproducibility and local experimentation only.

## Preprocessing plan

### Normalize regulation structure

Flatten each regulation into normalized units with fields such as:

- `frbr_uri`
- `type`
- `number`
- `year`
- `title`
- `status`
- `node_type`
- `node_number`
- `heading`
- `content`
- `parent_id`
- `sort_order`

### Create derived QA units

For each regulation, derive candidate question-answer pairs from:

- title
- chapter heading
- article heading
- article content
- legal relations if useful

### Data cleaning rules

- remove empty content nodes
- merge short fragmented nodes when needed
- preserve article numbers and regulation references
- keep the source reference alongside every derived sample

## Annotation plan

### Training data

Training data may be semi-automatic.

Possible sources:

- question templates
- synthetic questions generated from article content
- manually revised synthetic answers where needed

### Validation and test data

Validation and test should be curated more carefully.

Recommended approach:

- manually review all validation samples
- manually review all test samples
- ensure paraphrased questions are included
- avoid verbatim copying from article text whenever possible

This is critical to prevent the model from looking better than it really is.

## Dataset format for this repo

This repository already expects simple JSONL records with a `text` field.

### Training format option A: document-grounded instruction format

```json
{"text":"Dokumen: <potongan dokumen>\nPertanyaan: Apa definisi data pribadi?\nJawaban: Data pribadi adalah ..."}
```

### Evaluation format option B: document-free inference format

```json
{"text":"Pertanyaan: Apa definisi data pribadi?\nJawaban: Data pribadi adalah ..."}
```

### Recommended implementation

- train with document-grounded samples
- evaluate with document-free samples

This directly tests internalization into adapter parameters.

## Train/validation/test split strategy

### Do not split randomly by sample only

If the same regulation appears in both train and test, the model may simply memorize local passages.

### Recommended split

- split by regulation or by regulation family
- keep validation and test regulations separate from train regulations where possible

If a full held-out-document split is too harsh for the first iteration, use two evaluation modes:

- in-domain held-out questions from seen regulations
- out-of-domain held-out regulations

This gives a stronger thesis story.

## Model plan

### Baseline models

- `TinyLlama + LoRA`
- `Qwen3 4B 8bit + QLoRA`

### Optional comparison models

- `Mistral q4 + QLoRA`

### Recommendation

- use `TinyLlama` as the lightweight baseline
- use `Qwen3 4B 8bit` as the strongest current candidate for the main thesis run
- avoid relying on `Gemma 4` for the main thesis until there is a stronger reason to revisit it

## Experimental conditions

The core design uses three main conditions and one optional supplemental condition.

### A. Base model without document context at inference

- input contains the question only
- this is the no-context baseline

### B. Base model with source document context at inference

- input contains the question plus the relevant document or excerpt
- this measures the value of explicit context injection

### C. Base model plus LoRA or QLoRA adapter without document context at inference

- input contains the question only
- adapter carries the internalized document knowledge
- this is the main proposed condition

### D. Adapter model with source document context at inference

- optional supplemental condition
- useful if the thesis later wants to test whether adapters and explicit context are complementary

### Interpretation of comparisons

- `A vs B` measures the benefit of document context at inference time
- `B vs C` measures how close adapter internalization comes to the context-based approach
- `A vs C` measures the value of the proposed adapter-based method over the plain no-context baseline

## Evaluation plan

### Automatic metrics

- `EM`
- token-level `F1`
- optional `ROUGE-L`
- optional semantic similarity scoring

### Experimental reading dimensions

In the thesis narrative, automatic scores should be read together with:

- answer quality
- factual consistency
- evidence support
- source traceability
- inference efficiency

### Task-specific metrics

- citation accuracy
- article-number accuracy
- fact coverage for answer rubrics

### Manual evaluation

Include a human evaluation rubric for a subset.

Suggested rubric dimensions:

- substantive correctness
- completeness
- hallucination level
- citation correctness if applicable
- clarity of answer
- evidence support from the source regulation
- traceability back to the originating document or article

### Efficiency measurements

Because the design explicitly compares document-context inference with adapter-based inference, collect simple efficiency indicators such as:

- prompt length in tokens
- average inference latency per example
- tokens generated per second if available
- qualitative memory and operational complexity differences between condition B and condition C

## Risk analysis

### Methodological risks

- train/test leakage through the same regulation
- superficial memorization of legal phrasing
- synthetic question bias
- overclaiming parametric knowledge as legal understanding

### Operational risks

- rate-limit violations
- accidental key leakage from `.env`
- excessive dependence on Pasal.id structure instead of official-source replication

### Legal and publication risks

- publishing too much structured derived data from Pasal.id
- redistributing raw API responses

## Implementation plan in this repo

### Phase 1: ingestion

Add a script such as:

```text
scripts/build_pasalid_corpus.py
```

Responsibilities:

- load `PASAL_ID_API_KEY` from `.env`
- call `/api/v1/laws`
- call `/api/v1/laws/{frbr_uri}`
- save local raw cache

### Phase 2: dataset construction

Add a script such as:

```text
scripts/build_pasalid_qa_dataset.py
```

Responsibilities:

- flatten regulation content
- generate training examples
- create `train.jsonl`, `valid.jsonl`, `test.jsonl`

Suggested output:

```text
data/pasalid/train.jsonl
data/pasalid/valid.jsonl
data/pasalid/test.jsonl
```

### Phase 3: configs

Add config presets such as:

```text
configs/pasalid_tinyllama.yaml
configs/pasalid_qwen3_4b_8bit.yaml
```

### Phase 4: evaluation support

Either extend the current evaluation flow or add a dedicated legal evaluator:

```text
src/lora_mlx/eval_legal.py
```

This evaluator can support:

- relaxed answer normalization
- citation parsing
- rubric-friendly exported outputs

### Phase 5: thesis reporting

Add a dedicated report file such as:

```text
docs/pasalid-thesis-experiment-report.md
```

## Immediate next steps

1. Confirm the first legal domain subset to ingest
2. Add environment-variable loading for `PASAL_ID_API_KEY`
3. Build the Pasal.id ingestion script
4. Build a small pilot dataset with `20` to `100` manually reviewed QA pairs
5. Run a first baseline experiment with `TinyLlama` and `Qwen3`

## Success criteria for the first milestone

- local ingestion from Pasal.id works with the API key from `.env`
- a clean pilot JSONL dataset exists in `data/pasalid/`
- base versus adapter evaluation works without source documents at inference
- the report can show whether the adapter captures meaningful document knowledge beyond the base model
