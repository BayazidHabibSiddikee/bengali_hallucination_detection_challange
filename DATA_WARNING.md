# ⚠️ DO NOT MODIFY COMPETITION DATA

## Important Notice

**The `data/` and `dev_data/` directories contain official competition datasets from the ICT Fest Datathon 2026.**

### ❌ What NOT to do:
- ❌ Do NOT modify, delete, or rename files in `data/` or `dev_data/`
- ❌ Do NOT add or combine external data into these directories
- ❌ Do NOT create derived datasets in these locations
- ❌ Do NOT alter any CSV files or metadata

### ✅ What you SHOULD do instead:
- ✅ Create a **new subdirectory** for your processed data (e.g., `processed_data/`, `features/`, `artifacts/`)
- ✅ Keep original data **read-only** for reproducibility
- ✅ Document all external data usage separately with proper citations
- ✅ Track your data transformations in code (notebooks or scripts)

### Example Structure:
```
bengali_hallucination_detection/
├── data/                      ← Original competition data (READ-ONLY)
├── dev_data/                  ← Official dev set (READ-ONLY)
├── processed_data/            ← Your derived datasets
├── features/                  ← Feature engineering outputs
├── submissions/               ← Final submission CSVs
└── notebooks/                 ← Kaggle notebooks and analysis
```

### Phase 2 Compliance:
Per the rulebook, **all Phase 2 submissions must be reproducible** with the original competition data. Modifying the data directories will break reproducibility and result in disqualification.

---

**Last updated:** July 8, 2026  
**Competition:** অলীকবচন: Bengali LLM Hallucination Detection Challenge @ IUT 12th ICT Fest 2026
