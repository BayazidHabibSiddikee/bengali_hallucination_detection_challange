# 🔍 অলীকবচন — Bengali LLM Hallucination Detection

> **Competition:** IUT 12th ICT Fest Datathon 2026 — BrainLab  
> **Task:** Detect whether a Bengali LLM response is Faithful (`label=1`) or Hallucinated (`label=0`)  
> **Metric:** Binary F1 on the **Hallucinated class** (`label=0`)  
> **Tiebreaker:** F1 on **C1 cultural-distance subset** (Bengali vocabulary, idioms, history)  
> **Status:** ✅ Phase 1 complete — pipeline frozen, checkpoints uploaded

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INPUT ROW                                         │
│  { prompt_bn, context (optional), response_bn }                          │
└─────────────────────────┬───────────────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  REGIME     │
                    │  ROUTER     │
                    └──────┬──────┘
               ┌───────────┴────────────┐
         HAS CONTEXT?               NO CONTEXT?
         (130/299 val)              (169/299 val)
               │                        │
         ┌─────▼─────┐           ┌──────▼──────┐
         │ Compreh-  │           │   FAISS     │
         │ ension    │           │  Retriever  │
         │ Branch    │           │  (Wiki BN)  │
         └─────┬─────┘           └──────┬──────┘
               │                  Finds top-5 relevant
               │                  Wikipedia passages
               │                  → attaches as synthetic context
               └──────────┬─────────────┘
                           │
          ┌────────────────▼────────────────┐
          │     ENCODER ENSEMBLE (GPU)       │
          │  BanglaBERT-Large  (1.3 GB)     │
          │  mDeBERTa-v3-base  (1.1 GB)     │
          │  XLM-RoBERTa-large (1.7 GB)     │
          │       avg → enc signal           │
          └─────────────┬──────────────────┘
                        │
          ┌─────────────▼──────────────────┐
          │   NUCLEAR CLEAR (VRAM reset)    │
          │   Frees ~28 GB before LLM load  │
          └─────────────┬──────────────────┘
                        │
          ┌─────────────▼──────────────────┐
          │   TigerLLM-9B Judge (4-bit)     │
          │   md-nishat-008/TigerLLM-9B-it  │
          │   Category-aware system prompts  │
          │   → llm signal [0,1]            │
          └─────────────┬──────────────────┘
                        │
          ┌─────────────▼──────────────────┐
          │   LEXICAL + STATISTICAL         │
          │   lex, tfidf_sim, retr_sim      │
          │   prompt/ctx/resp lengths       │
          │   signal_skew/kurt/std/range    │
          │   n_signals_hallu, category_enc │
          └─────────────┬──────────────────┘
                        │
          ┌─────────────▼──────────────────┐
          │   Z-SCORE NORMALIZATION         │
          │   Fit on val set only           │
          │   (prevents test-set leakage)   │
          └─────────────┬──────────────────┘
                        │
          ┌─────────────▼──────────────────┐
          │   LightGBM META-STACKER        │
          │   Two models: has_ctx / no_ctx  │
          │   5-Fold StratifiedKFold OOF    │
          │   Platt calibration on OOF      │
          │   Threshold tuned via bootstrap │
          └─────────────┬──────────────────┘
                        │
          ┌─────────────▼──────────────────┐
          │       submission.csv            │
          │   { id, label }  2516 rows      │
          └────────────────────────────────┘
```

---

## 📁 Repository Structure

```
bengali_hallucination_detection/
├── pipeline.ipynb                   ← Main Kaggle submission notebook (21 cells)
├── adding_multifold_training.ipynb  ← Offline 5-fold encoder training (Kaggle)
├── pipeline_colab.ipynb             ← Colab variant (no GPU quota limits)
│
├── README.md                        ← This file
├── memory.md                        ← Detailed pipeline & memory management notes
├── architecture.md                  ← Extended architectural decisions
├── design.md                        ← Design philosophy and tradeoffs
├── phases.md                        ← Phase-by-phase execution plan (Phases 0–9)
├── prd.md                           ← Product requirements document
├── rules.md                         ← Competition rule interpretation
├── DATA_WARNING.md                  ← Data leakage risks & compliance checklist
├── STUDY_GUIDE.md                   ← Quick-start guide for new contributors
│
├── kernel-metadata.json             ← Kaggle dataset attachments
├── push_and_submit.sh               ← One-command push → submit
│
├── data/                            ← Competition data (gitignored)
├── comp_data/                       ← Raw competition files
├── real_data/                       ← Verified clean splits
├── dev_data/                        ← Local dev/test copies
│
├── banglabert_large.pt              ← Trained BanglaBERT-Large checkpoint
├── banglabert_large_pseudo.pt       ← BanglaBERT after pseudo-label round 2
├── mdeberta.pt                      ← Trained mDeBERTa-v3-base checkpoint
│
├── submission.csv                   ← Latest submission (2516 rows)
├── pseudo_labels.csv                ← High-confidence test predictions
├── val_signals.csv                  ← All 4 signals + meta for val set (299 rows)
├── test_signals.csv                 ← All 4 signals + meta for test set (2516 rows)
│
└── logs/                            ← Run logs
```

---

## 🗂️ Dataset Sources (`kernel-metadata.json`)

| Dataset | Role |
|---------|------|
| `bayazidhs/bengali-hallucination-data` | Competition files: `dataset samples.json`, `test set.csv` |
| `disisbig/bengali-wikipedia-articles` | 7,000+ Wikipedia articles → FAISS retrieval + cloze synthesis |
| `ajmainmahtab/bangla-natural-language-inference-dataset` | 8,400 NLI pairs for extra training signal |
| `mahdihasanqurishi/banglahallueval-qa` | Real Bengali QA hallucination pairs |
| `bayazidhs/bengali-historical-books` | History/cultural augmentation data |
| `bayazidhs/trained-banglabert` | Pre-trained `banglabert_large.pt` — **skips 3-hour training!** |
| `bayazidhs/tigerllm-9b-4bit` | Pre-quantized TigerLLM-9B weights |
| `bayazidhs/pseudo-labels` | `pseudo_labels.csv` from previous run — added to training data |

---

## 📓 Notebook Cell Map (`pipeline.ipynb` — 21 cells)

| Cell | Name | Purpose |
|------|------|---------|
| 0 | **INSTALLS** | Pin faiss-gpu, sentence-transformers, lightgbm, bitsandbytes |
| 1 | **CONFIG** | `CFG` dataclass, seeds, HF_TOKEN, paths |
| 2 | **BENGALI UTILS** | `is_no_ctx()`, `is_math_or_logic()`, Bengali digit normalization |
| 3 | **COMPETITION DATA** | Load `dataset samples.json` → `sample` (299 rows), `test set.csv` → `test` (2516 rows) |
| 4 | **NLI SOURCES** | Bangla NLI TSV + IndicXNLI-bn |
| 5 | **REAL QA** | BanglaHalluEval QA (1000 + full) + Bengali historical books |
| 6 | **CLOZE SYNTHETIC** | Generate synthetic hallucinations from Wikipedia passages |
| 7 | **ASSEMBLE** | Merge all data, mode-stratified 50/50 balance, `train_main` |
| 8 | **MODEL UTILS** | `PairDS`, `Focal` loss, `predict_proba()`, `train_backbone()` |
| 9 | **ENCODER ENSEMBLE** | Load/train BanglaBERT + mDeBERTa → `sig_val["enc"]`, `sig_test["enc"]` |
| 10 | **FAISS RETRIEVAL** | Dense embedding of Wiki, score no-ctx rows |
| 11 | **LEX/NUM** | Word overlap (lex), text lengths |
| 12 | **NUCLEAR CLEAR** | Destroy encoders, reclaim VRAM, compute LLM memory budget |
| 13 | **TIGERLLM JUDGE** | Category-aware 9B LLM judge → `llm_val`, `llm_test` |
| 14 | **META FEATURES** | `stackX()`, `add_meta_features()`, `z_score_norm()` |
| 15 | **LIGHTGBM** | Stratified 5-Fold OOF → Platt calibration → threshold → final model |
| 16 | **PSEUDO-LABEL** | Confident test predictions → retrain BanglaBERT → update signals |
| 17 | **SUBMISSION** | Write `submission.csv` |
| 18 | **DIAGNOSTICS** | Per-regime F1, save `val_signals.csv` + `test_signals.csv` |
| 19 | **EXPORT PSEUDO-LABELS** | Save `pseudo_labels.csv` for next run |
| 20 | **VISUALIZATIONS** | Error analysis charts |

---

## 🧠 LightGBM Feature Matrix

| Feature | Source | Description |
|---------|--------|-------------|
| `enc` | BanglaBERT + mDeBERTa average | Core NLI probability |
| `lex` | Lexical overlap | Word/char overlap between context and response |
| `retr` | FAISS + BanglaBERT re-rank | Retrieval score for no-context rows |
| `llm` | TigerLLM-9B (4-bit) | LLM faithfulness probability |
| `no_ctx` | Rule | Boolean: 0=has passage, 1=knowledge question |
| `prompt_len` | Text stats | Length of question in characters |
| `ctx_len` | Text stats | Length of context passage |
| `resp_len` | Text stats | Length of response |
| `tfidf_sim` | TF-IDF cosine | Char n-gram similarity (fit on val, transform on test) |
| `retr_sim` | FAISS cosine | Top-1 retrieval similarity score |
| `signal_skew` | Statistics | Bias toward one class across all 4 signals |
| `signal_kurt` | Statistics | Agreement tightness across all 4 signals |
| `signal_std` | Statistics | Raw disagreement between signals |
| `signal_range` | Statistics | Max span between highest/lowest signal |
| `signal_max_mean_gap` | Statistics | Outlier detector: is one signal very different? |
| `n_signals_hallu` | Voting | How many signals predict Hallucinated (< 0.5)? |
| `n_signals_missing` | Completeness | NaN count (no_ctx rows have more) |
| `category_enc` | Category router | 0=comprehension, 1=math, 2=vocabulary, 3=general, 4=history, 5=code_mixed |

---

## 🎯 Category Routing

| Category | Detection Rule | TigerLLM Role | Competition Relevance |
|----------|---------------|---------------|----------------------|
| `comprehension` | Has a real context passage | Passage grounding check | Most common (104/299 val) |
| `vocabulary` | `অর্থ, ভাবার্থ, বাগধারা, সমার্থক...` | Bengali linguist | **C1 tiebreaker** |
| `history` | `ইতিহাস, সাল, যুদ্ধ, মুক্তিযুদ্ধ...` | Bengali historian | High-value rows |
| `math` | `কত, যোগ, হিসাব, শতকরা...` | Math evaluator | Near-deterministic |
| `code_mixed` | Contains 2+ Latin chars | Multilingual analyst | Banglish/tech rows |
| `general_knowledge` | Fallback | Fact checker | 68/299 val |

---

## 🏗️ VRAM Budget (2× Kaggle T4, 15 GB each)

```
Phase 1 — Encoder Training/Loading
  GPU0: BanglaBERT-Large  ~1.3 GB
  GPU1: mDeBERTa-v3-base  ~1.1 GB
  → NUCLEAR CLEAR: del + gc.collect() + empty_cache() + malloc_trim(0)
  ✓ Both GPUs returned to ~15.4 GB free

Phase 2 — LLM Loading
  GPU0 + GPU1 + CPU offload
  TigerLLM-9B in 4-bit ≈ ~5.5 GB GPU + rest on CPU
  → del llm + gc + empty_cache()

Phase 3 — LightGBM
  CPU only — GPUs completely free
```

---

## 🔄 Iterative Improvement Loop

```
Kaggle Run → outputs:
  ├── val_signals.csv   (299 rows, all 4 signals + meta)
  ├── test_signals.csv  (2516 rows, same schema)
  └── pseudo_labels.csv (top-confident test predictions, conf ≥ 0.99)

OFFLINE ITERATION (no GPU, < 1 second per experiment):
  1. Download val_signals.csv + test_signals.csv
  2. Upload as "bayazidhs/bengali-extracted-signals" dataset
  3. New notebook with ONLY LightGBM code — no waiting for 3-hr TigerLLM
  4. Sweep: THR_SHIFT ∈ [-0.1, +0.15], num_leaves ∈ [3,5,7,10,15]

PSEUDO-LABEL LOOP:
  Run N → download pseudo_labels.csv
         → upload to bayazidhs/pseudo-labels (replace)
         → Run N+1 sees test distribution in BanglaBERT training
```

---

## 🐛 Critical Bugs Fixed

| Bug | Impact | Fix |
|-----|--------|-----|
| `rank_norm` mixing val+test stats | **+71% hallucination rate** — destroyed all probabilities | Removed entirely |
| `stackX` NaN bug | LLM + enc were NaN for LightGBM | Fixed column assignment |
| `z_score_norm` missing in Round 2 | Round 2 LightGBM trained on different scale | Added to pseudo retrain |
| HF_TOKEN not in `os.environ` | Rate-limit → Rust panic → infinite kernel restarts | `os.environ["HF_TOKEN"] = HF_TOKEN` |
| Hardcoded `== 2516` assertions | Crash on Phase 2 held-out fold | Made flexible |
| TF-IDF leakage | Vectorizer fitted on test data | Fit on val, transform on test |
| Brutal THR_SHIFT | Brittle manual threshold | Replaced with Platt calibration |
| Pseudo-label ignored in training | Round 2 retrain gained nothing | Fixed `adding_multifold_training.ipynb` |

---

## 📊 Performance Reference

| Run | Hallucination Rate in Submission | Notes |
|-----|----------------------------------|-------|
| Old pipeline (rank_norm bug) | **71.2%** — disaster | rank_norm distorted all probabilities |
| Fixed pipeline | **~45–50%** | Matches real distribution |
| Val set ground truth | **45.5%** (136/299) | Reference point |

---

## 🛠️ Quick Commands

```bash
# Push notebook to Kaggle and submit
bash push_and_submit.sh

# Or manually:
kaggle kernels push
kaggle kernels status bayazidhabibsiddikee/bengali-hallucination-pipeline

# Download outputs (submission.csv, signals, pseudo_labels)
kaggle kernels output bayazidhabibsiddikee/bengali-hallucination-pipeline

# Upload updated pseudo_labels to dataset
kaggle datasets version -p . -m "Round N pseudo-labels" --dir-mode tar
```

---

## 📅 Development Timeline

| Date | Milestone |
|------|-----------|
| July 14, 2026 | Initial pipeline — BanglaBERT + mDeBERTa + LightGBM |
| July 14, 2026 | Discovered & fixed `rank_norm` data leakage bug |
| July 14, 2026 | Fixed `stackX` NaN propagation |
| July 15, 2026 | Added Platt calibration replacing manual `THR_SHIFT` |
| July 15, 2026 | Fixed TF-IDF leakage (fit on val only) |
| July 16, 2026 | Added XLM-RoBERTa-large as 3rd backbone |
| July 16, 2026 | Added LLRD injection to Colab training script |
| July 16, 2026 | Pruned weak models — kept 3 strongest only |
| July 16, 2026 | Created dedicated `pipeline_colab.ipynb` |
| July 16, 2026 | Fixed pseudo-label ingestion in `adding_multifold_training.ipynb` |
| July 21, 2026 | **Project frozen** — Phase 1 submission complete |

---

## ⚠️ Rules & Compliance

### Forbidden (Disqualification)
- ❌ External APIs (OpenAI, Claude, Gemini, etc.)
- ❌ Manually labeling test rows
- ❌ More than 4 submissions per day
- ❌ Sharing code with other teams before Phase 1 closes
- ❌ Training directly on test-set ground truth

### Legal Strategies Used
- ✅ Pseudo-labeling: predicting on test, then training on those predictions
- ✅ Pre-trained public checkpoints uploaded as Kaggle datasets
- ✅ Public Bengali Wikipedia, NLI, QA datasets

---

## 🏁 Final State (Project Archived)

The pipeline is complete and frozen. Key artifacts for Phase 2:

- **`pipeline.ipynb`** — runnable Kaggle inference notebook, cold-runs in < 9 hours
- **`banglabert_large_pseudo.pt`** — best checkpoint (pseudo-label round 2)
- **`mdeberta.pt`** — mDeBERTa checkpoint
- **`submission.csv`** — Phase 1 final submission
- **`val_signals.csv`** + **`test_signals.csv`** — all signals for offline LightGBM tuning
- **`pseudo_labels.csv`** — high-confidence test predictions for next BanglaBERT retrain

For Phase 2 (onsite), refer to [`phases.md`](phases.md) (Phase 8 & 9) and [`design.md`](design.md).

---

*Last updated: July 21, 2026 | Pipeline version: 21-cell / Round-2 Pseudo-Label / Phase 1 Frozen*
