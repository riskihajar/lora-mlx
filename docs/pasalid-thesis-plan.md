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

- token-level `F1` as the main quality metric
- `EM` as a strict auxiliary metric only
- optional `ROUGE-L`
- optional semantic similarity scoring

Note on `EM`:

- `EM` should be kept for completeness and comparison, but not treated as the main proof metric.
- In this legal QA setting, a substantively correct answer may still receive `EM = 0` because of wording variation, answer ordering, or source-format differences.
- For that reason, the main reading of answer quality should rely more heavily on `F1` than on `EM`.

### Experimental reading dimensions

In the thesis narrative, automatic scores should be read together with:

- answer quality
- factual consistency
- evidence support
- source traceability
- inference efficiency

### Task-specific metrics

- evidence attribution
- evidence support rate
- unsupported answer rate
- citation accuracy or citation component accuracy
- article-number accuracy
- fact coverage for answer rubrics

Operational interpretation:

- `evidence attribution` measures whether the answer explicitly points to a supporting legal source
- `evidence support rate` measures how often the answer is actually supported by the source document
- `unsupported answer rate` measures how often the answer contains unsupported claims or hallucinated legal content
- `citation accuracy` or `citation component accuracy` measures whether the cited regulation, year, and article are correct

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
- latency `p50`
- latency `p95`
- tokens generated per second if available
- memory usage if it can be measured consistently
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

The implementation should stay script-first, reproducible, and runnable without assistant intervention. Every major step should be executable manually from the CLI, and interactive scripts should still support explicit command-line arguments for reproducibility.

### Phase 1: source ingestion

Add a script such as:

```text
scripts/build_pasalid_corpus.py
```

Responsibilities:

- load `PASAL_ID_API_KEY` from `.env`
- call `/api/v1/laws`
- call `/api/v1/laws/{frbr_uri}`
- save local raw cache

Current implementation status:

- initial ingestion script exists
- `.env`-backed token loading exists

### Phase 2: canonical document unit construction

Add or refine a script such as:

```text
scripts/build_pasalid_doc_units.py
```

Responsibilities:

- load local Pasal.id raw cache
- keep only usable and policy-safe derived fields
- normalize legal documents into canonical units
- preserve source traceability per unit

Suggested output:

```text
data/pasalid/doc_units.jsonl
```

Each canonical record should minimally include:

- `law_id`
- `frbr_uri`
- `title`
- `article_number`
- `question_type_candidate`
- `source_reference`
- `source_doc`

### Phase 3: synthetic QA generation

Add a script such as:

```text
scripts/generate_pasalid_qa.py
```

Responsibilities:

- load `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` from `.env`
- use the OpenAI SDK against the configured OpenAI-like endpoint
- generate question-answer pairs from canonical document units
- support both CLI arguments and interactive prompts
- support append versus overwrite modes
- save a canonical QA bank before any train/test split

Suggested output:

```text
data/pasalid/qa_bank.jsonl
```

Each QA bank record should include at least:

- `law_id`
- `frbr_uri`
- `question`
- `answer`
- `source_reference`
- `source_doc`
- `question_type`
- `difficulty`
- `generation_model`

Interactive mode should ask for:

- input file
- output file
- number of laws or units to process
- overwrite or append
- question generation mode
- questions per fact unit

### Phase 4: experiment split builder

Add a script such as:

```text
scripts/split_pasalid_experiment.py
```

Responsibilities:

- split the canonical QA bank into experiment-ready subsets
- support seen-document and unseen-document evaluation
- avoid leakage by splitting at the law level
- optionally reserve paraphrased questions for seen-document test conditions

Suggested output:

```text
data/pasalid/train.jsonl
data/pasalid/valid.jsonl
data/pasalid/test_seen.jsonl
data/pasalid/test_unseen.jsonl
data/pasalid/split_manifest.json
```

### Phase 5: training configs and wrappers

Add config presets such as:

```text
configs/pasalid_tinyllama.yaml
configs/pasalid_qwen3_4b_8bit.yaml
```

Add wrapper scripts such as:

```text
scripts/train_pasalid_tinyllama.sh
scripts/train_pasalid_qwen3.sh
scripts/eval_pasalid_tinyllama.sh
scripts/eval_pasalid_qwen3.sh
```

Current implementation status:

- initial Pasal.id training configs exist
- initial training and evaluation wrappers exist

### Phase 6: experiment evaluator

Either extend the current evaluation flow or add a dedicated legal evaluator:

```text
src/lora_mlx/eval_legal.py
```

This evaluator can support:

- relaxed answer normalization
- citation parsing
- rubric-friendly exported outputs

This evaluator should support the three main conditions:

- A: base model without document context
- B: base model with source document context
- C: base model plus adapter without document context

It should also emit outputs that make the following comparisons easy to inspect:

- `A vs B`
- `B vs C`
- `A vs C`

### Phase 7: experiment reporting

Add a dedicated report file such as:

```text
docs/pasalid-thesis-experiment-report.md
```

## Detailed implementation checklist

### Track 1: data ingestion and cleaning

- [x] Add `.env`-backed Pasal.id API client
- [x] Add initial corpus ingestion script
- [x] Build local raw cache in `data/pasalid_raw/`
- [x] Filter for `content_verified == true` in the pilot pipeline
- [x] Add basic OCR and footer cleanup
- [ ] Add stronger text-quality scoring for article content
- [ ] Add filtering for administrative or low-value regulations if needed
- [ ] Add logging summary for kept versus dropped laws and articles

### Track 2: canonical document units

- [ ] Introduce canonical `doc_units.jsonl` output
- [ ] Define stable schema for document units
- [ ] Include `law_id`, `frbr_uri`, `source_reference`, and `source_doc`
- [ ] Group units by legal fact type where possible
- [ ] Add per-unit metadata needed for QA generation

### Track 3: QA generation with OpenAI-like endpoint

- [ ] Add OpenAI SDK-based helper module
- [ ] Validate model and endpoint at script startup
- [ ] Add `scripts/generate_pasalid_qa.py`
- [ ] Support interactive mode with prompts
- [ ] Support non-interactive mode with CLI args
- [ ] Generate multiple paraphrased questions per fact unit
- [ ] Include source-backed answers with traceable references
- [ ] Save a canonical QA bank before splitting
- [ ] Save generation metadata for reproducibility

### Track 4: experimental split builder

- [ ] Split QA bank by law, not by random sample
- [ ] Create `train.jsonl`
- [ ] Create `valid.jsonl`
- [ ] Create `test_seen.jsonl`
- [ ] Create `test_unseen.jsonl`
- [ ] Create `split_manifest.json`
- [ ] Enforce that test questions are not literal duplicates of train questions
- [ ] Balance splits by question type where feasible
- [ ] Balance splits by answer length where feasible

### Track 5: baseline training

- [x] Add initial TinyLlama Pasal.id train wrapper
- [x] Add initial Qwen3 Pasal.id train wrapper
- [ ] Run TinyLlama baseline on cleaned split dataset
- [ ] Run Qwen3 baseline on cleaned split dataset
- [ ] Save adapters with experiment-specific names
- [ ] Record train and validation losses in a reportable format

### Track 6: A/B/C evaluation

- [ ] Implement evaluation prompts for condition A
- [ ] Implement evaluation prompts for condition B
- [ ] Implement evaluation prompts for condition C
- [ ] Export comparable outputs for the same question set
- [ ] Measure answer quality and token overlap metrics
- [ ] Measure factual consistency and source traceability manually on a subset
- [ ] Measure inference efficiency for B versus C

### Track 7: documentation and reproducibility

- [ ] Document every command needed to reproduce ingestion
- [ ] Document every command needed to reproduce QA generation
- [ ] Document every command needed to reproduce split building
- [ ] Document every command needed to reproduce training and evaluation
- [ ] Add run manifests for dataset and model artifacts
- [ ] Keep generated raw caches and dataset derivatives out of git where appropriate

## Immediate next steps

1. Build canonical document units from the current cleaned Pasal.id cache
2. Add OpenAI SDK-based QA generation as a standalone interactive script
3. Generate a first canonical QA bank from a small but clean legal subset
4. Build `train`, `valid`, `test_seen`, and `test_unseen` from that QA bank
5. Rerun the TinyLlama baseline on the split dataset before moving to Qwen3

## Success criteria for the first milestone

- local ingestion from Pasal.id works with the API key from `.env`
- a clean canonical QA bank exists in `data/pasalid/`
- experiment splits exist for `train`, `valid`, `test_seen`, and `test_unseen`
- condition A, B, and C can all be executed from scripts
- the report can show whether the adapter captures meaningful document knowledge beyond the base model and how close it gets to the context-based baseline
