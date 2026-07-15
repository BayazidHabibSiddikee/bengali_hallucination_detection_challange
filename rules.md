# Project Rules and Competition Guardrails

## 1. Authority
The official competition rulebook is the source of truth. If this file conflicts with an organizer announcement or updated rulebook, the organizer rule takes precedence.

## 2. Label Semantics
- `0` = hallucinated
- `1` = faithful
- Primary metric = binary F1 on the hallucinated class (`label = 0`).

Do not reverse these semantics in training, thresholding, evaluation, or submission code.

## 3. Official Data Is Immutable
The official `data/` and `dev_data/` content is read-only.

Never:
- edit, delete, rename, or overwrite official files;
- add external data into official-data directories;
- save transformed datasets beside the originals;
- manually alter test rows, IDs, ordering, or metadata.

Derived data belongs in dedicated locations such as:
- `processed_data/`
- `features/`
- `artifacts/`
- `outputs/`
- `submissions/`

## 4. Test-Set Isolation
The competition test set may be used only for inference.

Prohibited:
- fine-tuning on test examples;
- pseudo-labeling test examples and feeding them back into training;
- manually labeling or annotating test examples;
- using test-derived datasets as external training data;
- probing labels through systematic submissions;
- hardcoding test IDs, row order, row count, or test-specific values.

**Implementation rule:** set `pseudo_label_n = 0` and remove/disable all test pseudo-label retraining/export paths in the competition notebook.

## 5. External Data
External data is allowed only when it is:
- publicly available, publicly curated, or created during permitted competition runtime;
- unrelated to and not derived from the competition test set;
- clearly declared;
- properly cited;
- reproducibly obtainable or attached as a Kaggle dataset.

Maintain a data registry containing:
- dataset name;
- source;
- license/terms where available;
- purpose;
- preprocessing;
- version or snapshot date.

## 6. Models and APIs
Allowed:
- publicly available open-weight models;
- locally attached Kaggle/Hugging Face model assets;
- fine-tuning on permitted training data.

Not allowed:
- OpenAI API;
- Claude API;
- hosted inference endpoints;
- paid or external APIs during the competition solution;
- any model dependency that cannot be reproduced offline.

For safety and consistency, this project adopts the stricter interpretation: **no external API at any stage of the competition solution.**

## 7. Compute Constraints
The final inference package must:
- run in the standard permitted Kaggle environment;
- finish in under 9 hours;
- use no more than 50 GB total on-disk model weights;
- work on the organizer-specified P100 or dual-T4 setup.

A faster local development run does not prove compliance. Measure the final notebook in the target environment.

## 8. Reproducibility
Every scored solution must be reproducible from:
1. official raw inputs;
2. declared public external data;
3. declared model weights;
4. committed code/notebook;
5. fixed configuration.

Required:
- deterministic seeds where feasible;
- versioned dependencies;
- no hidden local files;
- no manual edits between notebook cells;
- no dependence on previous notebook state;
- no hardcoded predictions.

## 9. Submission Rules
The generated CSV must:
- contain exactly `id,label`;
- preserve official test IDs;
- contain one row per test example;
- contain only integer labels `0` or `1`;
- contain no missing values;
- match the sample-submission schema.

Do not exceed the competition submission limit.

## 10. Fair Play
Never:
- share private code, predictions, or models with other teams before the permitted deadline;
- bypass team submission limits using individual accounts;
- collude with another team;
- misrepresent authorship;
- plagiarize code, text, experiments, or ideas;
- privately exploit suspected dataset errors for advantage.

Public discussion and properly credited public resources are acceptable where competition rules permit them.

## 11. Engineering Rules

### R-1 — Configuration first
Paths, model IDs, thresholds, feature flags, and limits belong in configuration, not scattered magic constants.

### R-2 — No silent critical failure
Critical data/model/inference failures must raise clear errors. Optional external-data sources may be skipped only with an explicit log message.

### R-3 — Validate every boundary
Validate:
- input schemas;
- label values;
- model output shapes;
- NaN rates;
- row alignment;
- submission schema.

### R-4 — Separate training and inference
The Phase 2 inference notebook must not require training. Training may live in a separate notebook or script.

### R-5 — Preserve provenance
Every artifact must record the configuration and source version that produced it.

### R-6 — Avoid leakage
Never tune thresholds, model weights, retrieval parameters, or feature logic using hidden/test labels or leaderboard probing.

### R-7 — Dynamic dataset assumptions
Do not hardcode:
- test length;
- test row order beyond preserving the loaded order;
- specific IDs;
- values unique to the current test file.

### R-8 — One canonical implementation
Documentation must describe the code that actually runs. Update stale references immediately.

## 12. Current Prototype Changes Required Before Submission
1. Disable test pseudo-label retraining and export.
2. Replace the hardcoded `len(out) == 2516` assertion with `len(out) == len(test) == len(sub)`.
3. Align README/model documentation with the actual backbone configuration.
4. Replace the outdated Powell-optimization documentation with LightGBM stacking documentation.
5. Audit every external dataset and model for public availability and citations.
6. Benchmark the exact final inference notebook under the required Kaggle hardware/runtime.
