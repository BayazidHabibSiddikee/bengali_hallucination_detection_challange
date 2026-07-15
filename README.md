# 🔍 অলীকবচন — Bengali LLM Hallucination Detection Pipeline

> **Competition:** IUT 12th ICT Fest Datathon 2026 — BrainLab  
> **Task:** Detect whether a Bengali LLM response is Faithful (`label=1`) or Hallucinated (`label=0`)  
> **Metric:** Binary F1 on the **HALLUCINATED class** (`label=0`)  
> **Tiebreaker:** F1 on **C1 cultural-distance subset** (Bengali vocabulary, idioms, history)

---

## 📐 High-Level Architecture

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
               │                  → attaches as context
               └──────────┬─────────────┘
                           │
          ┌────────────────▼────────────────┐
          │     ENCODER ENSEMBLE (GPU)       │
          │  ┌──────────────────────────┐   │
          │  │  BanglaBERT-Large (1.3GB)│   │
          │  │  csebuetnlp/banglabert   │   │
          │  └──────────┬───────────────┘   │
          │  ┌──────────▼───────────────┐   │
          │  │  mDeBERTa-v3-base (1.1GB)│   │
          │  │  microsoft/mdeberta-v3   │   │
          │  └──────────┬───────────────┘   │
          │       avg → enc signal           │
          └─────────────┬──────────────────┘
                        │
          ┌─────────────▼──────────────────┐
          │   NUCLEAR CLEAR (VRAM reset)    │
          │   Frees ~28GB before LLM load   │
          └─────────────┬──────────────────┘
                        │
          ┌─────────────▼──────────────────┐
          │   TigerLLM-9B Judge (4-bit)     │
          │   md-nishat-008/TigerLLM-9B-it  │
          │   Category-aware system prompts  │
          │   ┌─────────────────────────┐   │
          │   │ comprehension           │   │
          │   │ vocabulary (C1)         │   │
          │   │ history    (C2)         │   │
          │   │ math                    │   │
          │   │ code_mixed              │   │
          │   │ general_knowledge       │   │
          │   └─────────────────────────┘   │
          │   → llm signal [0,1]            │
          └─────────────┬──────────────────┘
                        │
          ┌─────────────▼──────────────────┐
          │   LEXICAL + STATISTICAL         │
          │   lex, tfidf_sim, retr_sim      │
          │   prompt/ctx/resp lengths       │
          │   Cross-signal: skew, kurt, std │
          │   n_signals_hallu, category_enc │
          └─────────────┬──────────────────┘
                        │
          ┌─────────────▼──────────────────┐
          │   Z-SCORE NORMALIZATION         │
          │   Scale using val set only      │
          │   (prevents test-set leakage)   │
          └─────────────┬──────────────────┘
                        │
          ┌─────────────▼──────────────────┐
          │   LightGBM META-STACKER        │
          │   Two models: has_ctx / no_ctx  │
          │   5-Fold StratifiedKFold OOF    │
          │   Threshold tuned via bootstrap │
          │   THR_SHIFT = +0.05 applied     │
          └─────────────┬──────────────────┘
                        │
          ┌─────────────▼──────────────────┐
          │   PSEUDO-LABEL ROUND 2          │
          │   Top-500 confident test rows   │
          │   BanglaBERT retrained + repeat │
          └─────────────┬──────────────────┘
                        │
          ┌─────────────▼──────────────────┐
          │       submission.csv            │
          │   { id, label }  2516 rows      │
          └────────────────────────────────┘
```

---

## 📁 Notebook Cell Map (`pipeline.ipynb` — 21 cells)

| Cell | Name | Purpose |
|------|------|---------|
| 0 | **INSTALLS** | Pin faiss-gpu, sentence-transformers, lightgbm, bitsandbytes |
| 1 | **CONFIG** | `CFG` dataclass, seeds, HF_TOKEN, paths |
| 2 | **BENGALI UTILS** | `is_no_ctx()`, `is_math_or_logic()`, Bengali digit normalization |
| 3 | **COMPETITION DATA** | Load `dataset samples.json` → `sample (299 rows)`, load `test set.csv` → `test (2516 rows)` |
| 4 | **NLI SOURCES** | Load Bangla NLI TSV + IndicXNLI-bn for extra training signal |
| 5 | **REAL QA** | Load BanglaHalluEval QA datasets (1000 + full) + Bengali historical books |
| 6 | **CLOZE SYNTHETIC** | Generate synthetic hallucinations from Wikipedia passages |
| 7 | **ASSEMBLE** | Merge all data, mode-stratified balance (1:2 ratio), `train_main` |
| 8 | **MODEL UTILS** | `PairDS`, `Focal` loss, `predict_proba()`, `train_backbone()` |
| 9 | **ENCODER ENSEMBLE** | Load/train BanglaBERT + mDeBERTa → `sig_val["enc"]`, `sig_test["enc"]` |
| 10 | **FAISS RETRIEVAL** | Dense embedding of Wiki, score no-ctx rows via BanglaBERT passage re-rank |
| 11 | **LEX/NUM** | Word overlap (lex), text lengths |
| 12 | **NUCLEAR CLEAR** | Destroy encoders, reclaim VRAM, compute LLM memory budget |
| 13 | **TIGERLLM JUDGE** | Category-aware 9B LLM judge → `llm_val`, `llm_test` |
| 14 | **META FEATURES** | `stackX()`, `add_meta_features()`, `z_score_norm()` — builds LightGBM feature matrix |
| 15 | **LIGHTGBM** | Stratified 5-Fold OOF → threshold → final model → `pv`, `pt` |
| 16 | **PSEUDO-LABEL** | Confident test predictions → retrain BanglaBERT → update signals |
| 17 | **SUBMISSION** | Write `submission.csv` |
| 18 | **DIAGNOSTICS** | Per-regime F1, save `val_signals.csv` + `test_signals.csv` |
| 19 | **EXPORT PSEUDO-LABELS** | Save `pseudo_labels.csv` for next run |
| 20 | **VISUALIZATIONS** | Error analysis charts |

---

## 🧠 Feature Matrix Fed into LightGBM

| Feature | Source | Description |
|---------|--------|-------------|
| `enc` | BanglaBERT + mDeBERTa average | Core NLI probability |
| `lex` | Lexical overlap | Word/char overlap between context and response |
| `retr` | FAISS + BanglaBERT re-rank | Retrieval score for no-context rows |
| `llm` | TigerLLM-9B (4-bit) | LLM faithfulness probability |
| `no_ctx` | Rule | Boolean: 0=has passage, 1=knowledge question |
| `prompt_len` | Text stats | Length of the question in characters |
| `ctx_len` | Text stats | Length of the context passage |
| `resp_len` | Text stats | Length of the response |
| `tfidf_sim` | TF-IDF cosine | Char n-gram similarity between prompt and context |
| `retr_sim` | FAISS cosine | Top-1 retrieval similarity score |
| `signal_skew` | Statistics | Are the 4 signals biased toward one class? |
| `signal_kurt` | Statistics | How tightly do all 4 signals agree? |
| `signal_std` | Statistics | Raw disagreement between all 4 signals |
| `signal_range` | Statistics | Max span between highest and lowest signal |
| `signal_max_mean_gap` | Statistics | Is one signal an outlier vs the rest? |
| `n_signals_hallu` | Voting | How many signals predict Hallucinated (< 0.5)? |
| `n_signals_missing` | Completeness | How many signals are NaN? (no_ctx rows have more) |
| `category_enc` | Category router | 0=comprehension, 1=math, 2=vocabulary, 3=general, 4=history, 5=code_mixed |

---

## 🗂️ Dataset Sources (kernel-metadata.json)

| Dataset | Role |
|---------|------|
| `bayazidhs/bengali-hallucination-data` | Competition files: `dataset samples.json`, `test set.csv` |
| `disisbig/bengali-wikipedia-articles` | 7,000+ Wikipedia articles for FAISS retrieval + cloze synthesis |
| `ajmainmahtab/bangla-natural-language-inference-dataset` | 8,400 NLI pairs for extra training |
| `mahdihasanqurishi/banglahallueval-qa` | Real Bengali QA hallucination pairs |
| `bayazidhs/bengali-historical-books` | Augmentation source for history/cultural data |
| `bayazidhs/trained-banglabert` | Pre-trained `banglabert_large.pt` — **skips 3-hour training!** |
| `bayazidhs/tigerllm-9b-4bit` | Pre-quantized TigerLLM-9B weights |
| `bayazidhs/pseudo-labels` | `pseudo_labels.csv` from previous run — added to training data |

---

## 🐛 Critical Bugs Fixed (History)

| Bug | Impact | Fix Applied |
|-----|--------|------------|
| `rank_norm` mixing val+test | **+71% hallucination rate** — main score killer | Removed entirely |
| `stackX` NaN bug | LLM + enc signals were NaN for LightGBM | Fixed column assignment |
| `z_score_norm` missing in Round 2 | Round 2 LightGBM trained on different scale | Added to pseudo-label retrain block |
| HF_TOKEN not in `os.environ` | HuggingFace rate-limits → corrupted cache → Rust panic → infinite kernel restarts | `os.environ["HF_TOKEN"] = HF_TOKEN` |
| Hardcoded `==2516` assertions | Would crash on Phase 2 held-out fold | Made flexible |
| Duplicate `torch.save` | Wasted time | Removed duplicate |
| `pseudo_labels.csv` empty (32 bytes) | Pseudo-label round 2 was useless | Fixed by fixing rank_norm first |

---

## 🏗️ Memory Architecture (VRAM Management)

```
Kaggle: 2× T4 GPU × 15GB = 30GB VRAM

Phase 1: Encoder Training/Loading
  GPU0: BanglaBERT-Large (1.3GB)  ← ~1.3GB
  GPU1: mDeBERTa-v3-base (1.1GB)  ← ~1.1GB
  After inference → NUCLEAR CLEAR (del + gc + malloc_trim)
  ✓ GPUs returned to ~15.4GB free each

Phase 2: LLM Loading
  GPU0 + GPU1 + CPU offload
  TigerLLM-9B in 4-bit ≈ ~5.5GB on GPU, rest on CPU
  After inference → del llm + gc + empty_cache()

Phase 3: LightGBM (CPU only, GPUs completely free)
```

---

## 🔄 The Iterative Learning Loop (How to Improve Between Runs)

```
Run 1: Full pipeline → outputs:
  ├── val_signals.csv   (299 rows, all 4 signals + meta features)
  ├── test_signals.csv  (2516 rows, same schema)
  └── pseudo_labels.csv (confident test predictions)

OFFLINE (no GPU needed, runs in <1 second):
  1. Download val_signals.csv + test_signals.csv
  2. Upload to Kaggle as "bayazidhs/bengali-extracted-signals"
  3. Create new notebook with ONLY LightGBM code
  4. Experiment: THR_SHIFT, num_leaves, feature engineering
  5. Submit that notebook (no waiting for 3hr TigerLLM!)

Run 2: Attach pseudo_labels.csv → BanglaBERT sees test distribution
Run 3: Attach pseudo_labels_round2.csv → ...
```

---

## 🎯 Competition Category Routing

The pipeline auto-detects the question type and gives TigerLLM a specialized system prompt:

| Category | Detection Rule | TigerLLM Role | Competition Relevance |
|----------|---------------|---------------|----------------------|
| `comprehension` | Has a real context passage | Passage grounding check | Most common (104/299 val) |
| `vocabulary` | `অর্থ, ভাবার্থ, বাগধারা, সমার্থক...` | Bengali linguist | **C1 tiebreaker** |
| `history` | `ইতিহাস, সাল, যুদ্ধ, মুক্তিযুদ্ধ...` | Bengali historian | High-value rows |
| `math` | `কত, যোগ, হিসাব, শতকরা...` | Math evaluator | Deterministic |
| `code_mixed` | Contains 2+ Latin chars | Multilingual analyst | Banglish/tech rows |
| `general_knowledge` | Fallback | Fact checker | 68/299 val |

---

## 🚀 Where to Work More (Priority Order)

### 🔴 HIGH IMPACT — Do These First

**1. Offline LightGBM Tuning (Free, Instant)**
- Download `test_signals.csv` and `val_signals.csv` from your Kaggle output
- Upload as a Kaggle dataset
- Create a clean notebook that ONLY runs LightGBM — no GPU needed
- Test dozens of `THR_SHIFT` values: `-0.1, -0.05, 0, +0.05, +0.1, +0.15`
- Test `num_leaves`: 3, 5, 7, 10, 15
- Each experiment takes < 1 second → can run 500 experiments in 10 minutes

**2. Pseudo-Label Quality Loop**
- After your next run, download `pseudo_labels.csv`
- Upload it to your `bayazidhs/pseudo-labels` dataset (replace the old one)
- Re-run the notebook — BanglaBERT will train on this extra data
- Repeat 2-3 times for compounding gains

**3. Per-Category Threshold Tuning**
- Currently one threshold for `has_ctx` and one for `no_ctx`  
- Better: separate thresholds for `comprehension`, `vocabulary`, `math`, `general_knowledge`
- Math rows are near-deterministic → lower threshold for label=0
- Vocabulary rows (C1 tiebreaker) → tune specifically

---

### 🟡 MEDIUM IMPACT — If You Have Time

**4. Expand the Bengali Knowledge Base**
- The pipeline's weakest point is `general_knowledge` no-ctx rows
- Add more factual Bengali knowledge to `wiki_passages` (currently only 7,091 passages)
- Consider adding Bengali textbook data or encyclopedia data as a new Kaggle dataset

**5. TigerLLM Prompt Tuning**
- The few-shot examples in `build_sys_prompt()` are currently generic
- For the **C1 tiebreaker**, add 2-3 vocabulary/idiom examples to the `vocabulary` category prompt
- For `history`, add examples involving common mistakes (wrong dates, wrong people)

**6. Ensemble Diversity**
- Currently: BanglaBERT + mDeBERTa averaged equally
- Better: try `xlm-roberta-large` instead of mDeBERTa (stronger on Bengali, ~1.7x slower to train)
- If checkpoints exist: weighted average (`0.6 × banglabert + 0.4 × mdeberta`) instead of `0.5 + 0.5`

---

### 🟢 LOW IMPACT — Polish / Phase 2 Prep

**7. Error Analysis (Cell 20)**
- Run the full visualization cell to see which specific rows you're getting wrong
- Look for patterns: Are all wrong rows from the same category? Same hallucination mode?
- Focus data augmentation on those specific failure patterns

**8. Report / Paper Writing**
- Phase 2 is 80% of your score — the paper and presentation matter enormously
- Start documenting: what failed (rank_norm), what worked (LLM judge), what you tried
- Use the architecture diagram in this README as a figure in your paper

**9. Code Cleanliness for Phase 2**
- Organize the notebook: add markdown cells explaining each stage
- Add proper docstrings to `run_llm_judge()`, `build_retr_signal()`, `add_meta_features()`
- The organizers will read and run your notebook — make it presentable

---

## ⚠️ Be Careful About These

### Things That Will Break Silently

| Risk | Description | Defense |
|------|-------------|---------|
| **Data leakage** | Computing stats on val+test together | Always fit scalers/normalizers on val only |
| **Threshold overfitting** | Tuning threshold on tiny 299-row val set | Use StratifiedKFold OOF (already done) |
| **Pseudo-label noise** | Low-confidence pseudo-labels hurt more than help | Keep `pseudo_conf >= 0.99` |
| **Category misclassification** | A math question routed to `vocabulary` gets wrong prompt | Audit `get_category()` on your val set |
| **Phase 2 row count** | Phase 2 has a different number of rows | No hardcoded `==2516` (already fixed) |

### Things That Are Forbidden (Read the Rules!)

- ❌ Using external APIs (OpenAI, Claude, etc.) — **disqualification**
- ❌ Manually labeling test rows — **disqualification**  
- ❌ Submitting > 4 times per day — **limit enforced**
- ❌ Sharing code with other teams before Phase 1 closes
- ❌ Fine-tuning on the test set (pseudo-labeling is OK, test-set training is not)

> ⚠️ **Note:** The pseudo-label strategy uploads already-predicted labels, not raw test data. This is legal. But training BanglaBERT directly on test-set ground truth (if you somehow knew it) would be disqualification.

---

## 📊 Current Performance Reference

| Run | Hallucination Rate in Submission | Notes |
|-----|----------------------------------|-------|
| Old (rank_norm bug) | **71.2%** (disaster) | rank_norm distorted all probabilities |
| Fixed pipeline (target) | **~45-50%** | Matches real distribution |
| Val set ground truth | 45.5% (136/299) | Reference point |

---

## 🛠️ Quick Commands

```bash
# Push notebook to Kaggle
kaggle kernels push

# Check submission status
kaggle kernels status bayazidhabibsiddikee/bengali-hallucination-pipeline

# Download outputs (submission.csv, signals, pseudo_labels)
kaggle kernels output bayazidhabibsiddikee/bengali-hallucination-pipeline

# Upload updated pseudo_labels to dataset
kaggle datasets version -p . -m "Round 2 pseudo-labels" --dir-mode tar
```

---

*Last updated: July 14, 2026 | Pipeline version: 21-cell / Round-2 Pseudo-Label*
