# Product Requirements Document (PRD)

## 1. Project
**Name:** Bengali Hallucination Detection Pipeline  
**Competition:** অলীকবচন: Bengali LLM Hallucination Detection Challenge — IUT 12th ICT Fest Datathon 2026  
**Primary objective:** Build a reproducible, offline inference system that classifies each Bengali LLM response as:

- `0` — hallucinated
- `1` — faithful

The optimization target is **binary F1 on the hallucinated class (`label = 0`)**.

## 2. Problem Statement
Given:
- a Bengali prompt,
- a candidate Bengali response,
- and optional supporting context,

the system must determine whether the response is faithful to the available evidence or hallucinated.

The project must support two regimes:
1. **Context-present:** judge faithfulness directly against the supplied context.
2. **No-context:** retrieve relevant public Bengali evidence locally, then judge the response.

## 3. Goals

### G1 — Competition performance
Maximize validation and held-out binary F1 for class `0`, while avoiding leaderboard-specific overfitting.

### G2 — Reproducibility
A fresh Kaggle notebook run must reproduce predictions from raw competition inputs and declared public model/data assets.

### G3 — Runtime compliance
Inference must complete within the competition's Kaggle runtime limit on a single P100 or dual T4 environment.

### G4 — Robustness
The pipeline must degrade gracefully when a large model cannot load, a non-critical external dataset is unavailable, or GPU memory is constrained.

### G5 — Explainability and auditability
Every prediction signal, dataset source, model dependency, transformation, and final output must be traceable.

## 4. Non-Goals
- Building a conversational chatbot.
- Generating corrected answers.
- Using hosted inference APIs.
- Manually labeling or inspecting test examples to improve predictions.
- Training on, pseudo-labeling from, or otherwise deriving training data from the competition test set.
- Optimizing solely for the public leaderboard.

## 5. Users and Stakeholders
- **Primary users:** competition team members.
- **Reviewers:** competition organizers and judges.
- **Operational environment:** Kaggle notebook runtime.
- **Secondary audience:** readers of the final paper, README, and presentation.

## 6. Functional Requirements

### FR-1 — Competition data ingestion
The system shall:
- load the official validation/sample, test, and sample-submission files;
- preserve official data unchanged;
- validate required columns and row counts dynamically where possible;
- normalize missing-context values consistently;
- preserve test `id` values and row order.

### FR-2 — Bengali text normalization
The system shall provide deterministic utilities for:
- Unicode NFC normalization;
- Bengali/ASCII digit normalization;
- tokenization for lexical features;
- numeric extraction;
- context-presence detection.

### FR-3 — Training-data construction
The system may construct training data from declared public sources such as:
- Bengali QA datasets;
- Bengali NLI datasets;
- public Bengali Wikipedia or other properly cited public corpora;
- synthetic transformations created from permitted public data.

Every source must be documented with provenance and licensing/availability notes.

### FR-4 — Encoder signal
The system shall support one or more Bengali/multilingual sequence-classification backbones. The current notebook configuration uses:
- `csebuetnlp/banglabert_large`
- `microsoft/mdeberta-v3-base`

The encoder subsystem shall output a probability of `label = 1` and may ensemble multiple backbones.

### FR-5 — Retrieval signal
For no-context examples, the system shall:
- build a local retrieval corpus from declared public data;
- create dense embeddings;
- retrieve top-k passages using FAISS;
- score the candidate response against retrieved evidence;
- expose retrieval similarity as a meta-feature.

### FR-6 — LLM judge signal
The system may use an open-weight local LLM to produce a classification signal without text generation. It shall:
- run fully offline during competition inference;
- prefer pre-attached model weights;
- use deterministic prompting;
- expose a scalar probability/logit-derived signal;
- fall back to a smaller local model if the primary model cannot load.

### FR-7 — Lexical and metadata features
The system shall compute lightweight features including, where applicable:
- lexical overlap;
- numeric consistency;
- prompt/context/response length;
- TF-IDF similarity;
- retrieval similarity.

### FR-8 — Meta-model
The system shall combine model signals and metadata using a validation-trained stacker. The current implementation uses separate LightGBM models for:
- context-present examples;
- no-context examples.

Thresholds shall be tuned only on permitted labeled validation/training data.

### FR-9 — Submission generation
The system shall output exactly:
- `id`
- `label`

Labels must be integer `0` or `1`, with no missing rows or extra columns.

### FR-10 — Diagnostics
The pipeline shall produce reproducibility and analysis artifacts such as:
- validation predictions/signals;
- error-analysis CSV;
- feature importance;
- runtime/memory logs;
- final submission CSV.

## 7. Quality Attributes

### Reproducibility
- fixed random seeds;
- pinned critical package versions;
- explicit model/data paths;
- no hidden manual steps;
- no dependence on undeclared local files.

### Reliability
- validate schemas before training/inference;
- fail fast on submission-format errors;
- use controlled fallback behavior;
- avoid silent exception swallowing for critical stages.

### Performance
- inference under 9 hours;
- total model weights under 50 GB;
- bounded retrieval corpus and sequence lengths;
- memory cleanup between heavy stages.

### Maintainability
The final competition notebook should be decomposable into modules for:
- config;
- data loading;
- preprocessing;
- augmentation;
- models;
- retrieval;
- features;
- stacking;
- evaluation;
- submission.

## 8. Success Metrics
1. Primary: binary F1 for `label = 0`.
2. Secondary:
   - context-present F1;
   - no-context F1;
   - reproducibility success rate;
   - total inference runtime;
   - peak GPU/CPU memory;
   - percentage of predictions produced without fallback/error.

## 9. Acceptance Criteria
The project is release-ready when:
- the inference notebook runs end-to-end from raw official test data;
- no prohibited API or test-derived training data is used;
- all external assets are declared and publicly accessible;
- runtime is below 9 hours in the target Kaggle environment;
- total attached model weights are below 50 GB;
- submission schema is valid;
- the final notebook contains clear documentation;
- the approach, data provenance, evaluation, and limitations are documented for the Phase 2 report.

## 10. Known Gaps in the Current Prototype
- The README describes XLM-RoBERTa-Large, while the current notebook config uses mDeBERTa-v3-base.
- The study guide describes Powell optimization, while the current notebook uses LightGBM stacking.
- The notebook contains test-set pseudo-label generation/retraining logic. This must be disabled or removed for the competition-compliant pipeline because the rules prohibit fine-tuning on the test set and test-derived external data.
- The submission code hardcodes `2516` rows. Replace this with validation against the loaded test/sample-submission length.
- Some broad exception handlers should be replaced with explicit errors and structured logging.
