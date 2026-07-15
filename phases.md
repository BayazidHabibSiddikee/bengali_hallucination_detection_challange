# Project Phases and Execution Plan

## Phase 0 — Compliance Freeze
**Goal:** establish a competition-safe baseline before further optimization.

### Tasks
- Read and freeze the current rule interpretation.
- Disable all test-set pseudo-label training/export.
- Inventory every external dataset and model.
- Verify all assets are public, declared, and reproducible.
- Remove external API dependencies.
- Make official data read-only.

### Exit criteria
- `rules.md` is accepted by the team.
- No test-derived training path is reachable.
- External asset registry is complete.

---

## Phase 1 — Reproducible Baseline
**Goal:** create the simplest trustworthy end-to-end classifier.

### Tasks
- Load official train/validation/test inputs.
- Normalize Bengali text and missing context.
- Train or load one encoder classifier.
- Evaluate binary F1 on class `0`.
- Tune threshold on permitted validation data.
- Generate a schema-valid submission.

### Deliverables
- baseline validation score;
- baseline submission;
- runtime and memory measurements;
- deterministic config snapshot.

### Exit criteria
- clean end-to-end run;
- no row-alignment failures;
- reproducible score within expected numerical tolerance.

---

## Phase 2 — Data and Evaluation Foundation
**Goal:** improve signal quality without leakage.

### Tasks
- Add permitted Bengali QA/NLI sources.
- Add synthetic hallucination transformations.
- Track source and transformation provenance.
- Build stratified evaluation by:
  - context availability;
  - source/domain;
  - hallucination type;
  - response length;
  - numeric content.
- Add error taxonomy.

### Exit criteria
- every training row has provenance;
- validation slices are reported;
- augmentation improves robust validation rather than only aggregate score.

---

## Phase 3 — Encoder Ensemble
**Goal:** build a strong discriminative core.

### Tasks
- Train/load BanglaBERT-Large.
- Train/load the selected multilingual backbone.
- Compare single-model and ensemble performance.
- Calibrate probability behavior.
- Save immutable checkpoints outside the inference notebook.

### Decision gate
Keep an additional backbone only if its incremental F1 gain justifies runtime and model-size cost.

### Exit criteria
- selected encoder set is frozen;
- inference checkpoints are attached and load successfully;
- training is not required for Phase 2 inference.

---

## Phase 4 — Retrieval for No-Context Examples
**Goal:** improve examples that lack supporting context.

### Tasks
- Build a public Bengali retrieval corpus.
- Chunk and embed the corpus.
- Build FAISS index.
- Retrieve top-k evidence.
- Score response faithfulness against retrieved evidence.
- Measure retrieval quality and downstream F1.

### Decision gate
Retain retrieval only if it improves no-context validation and remains within runtime/memory limits.

### Exit criteria
- retrieval corpus provenance documented;
- retrieval parameters frozen;
- no test-specific retrieval logic.

---

## Phase 5 — Local LLM Judge
**Goal:** add a complementary reasoning signal.

### Tasks
- Run an open-weight LLM locally.
- Use classification/logit extraction rather than free-form generation.
- Benchmark TigerLLM and permitted fallbacks.
- Verify prompt/token-label mapping.
- Measure incremental value over encoders and retrieval.

### Decision gate
Keep the LLM judge only if the F1 gain is worth its runtime and operational risk.

### Exit criteria
- fully offline inference;
- deterministic prompt format;
- tested fallback chain;
- measured runtime.

---

## Phase 6 — Meta-Model and Thresholding
**Goal:** combine complementary signals without overfitting.

### Inputs
- encoder ensemble probability;
- lexical consistency;
- retrieval score;
- LLM judge score;
- prompt/context/response lengths;
- TF-IDF similarity;
- retrieval similarity.

### Tasks
- train separate context/no-context LightGBM stackers;
- use leakage-safe validation;
- tune class-0 decision thresholds on permitted validation data;
- compare against simple averaging;
- inspect feature importance and ablations.

### Exit criteria
- stacker beats simpler baselines robustly;
- thresholds are frozen before final test inference;
- no leaderboard-driven parameter tuning.

---

## Phase 7 — Competition Inference Hardening
**Goal:** produce the Phase 2-ready inference package.

### Tasks
- split training and inference workflows;
- load pre-trained checkpoints only;
- remove dead experimental code;
- remove pseudo-label code;
- remove hardcoded test size;
- validate model-size budget;
- benchmark under target Kaggle hardware;
- test cold-start execution from top to bottom;
- save only required outputs.

### Required outputs
- `submission.csv`
- `errors.csv` or validation diagnostics, when labels are available
- `val_signals.csv`
- runtime/config log

### Exit criteria
- under 9 hours;
- under 50 GB model weights;
- no internet/API requirement;
- cold run succeeds.

---

## Phase 8 — Phase 2 Submission Package
**Goal:** prepare organizer-reviewable artifacts.

### Package
- runnable Kaggle inference notebook;
- optional training notebook;
- model checkpoints/weights;
- clear README/documentation;
- four-page paper excluding references;
- presentation slides.

### Paper structure
1. Problem and motivation
2. Data and preprocessing
3. Methodology
4. Experiments and ablations
5. Error analysis
6. Limitations
7. Reproducibility
8. References

### Exit criteria
- another team member can reproduce the package without oral instructions;
- every external asset is cited;
- paper claims match measured experiments.

---

## Phase 9 — Onsite Final Preparation
**Goal:** communicate the work clearly and defensibly.

### Prepare
- architecture diagram;
- experiment table;
- ablation results;
- error examples from permitted labeled data;
- runtime/model-size table;
- novelty statement;
- limitations and failure modes;
- Q&A answers on data, preprocessing, methodology, evaluation, and novelty.

### Final rule
Do not introduce an untested last-minute model change after the reproducible package is frozen.
