#!/usr/bin/env python3
"""Final audit fixes for pipeline.ipynb:
1. faiss: "faiss-gpu" pip wheel doesn't exist on Kaggle python -> ensure faiss-cpu
2. honest validation: fit_lgbm also returns its OOF array; pv is built from OOF
   (was: refit-model in-sample predictions -> inflated F1 prints + errors.csv)
   Applied to both CELL 15 and the CELL 15.5 pseudo-retrain re-stack.
3. round-2 pseudo retrain: cap epochs at 2 to protect the 9h kernel budget
"""
import json

NB = "pipeline.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]

def find_cell(marker):
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            return i
    raise SystemExit(f"marker not found: {marker}")

def repl(i, old, new, what):
    s = "".join(cells[i]["source"])
    assert old in s, f"anchor missing for: {what}"
    cells[i]["source"] = [s.replace(old, new)]
    print("fixed:", what)

# ---- 1. robust faiss install --------------------------------------------------
repl(find_cell("CELL 1 — INSTALLS"),
'''faiss_pkg = "faiss-gpu" if shutil.which("nvidia-smi") else "faiss-cpu"
subprocess.run([sys.executable,"-m","pip","install","-q",faiss_pkg],check=False)
print("ok | faiss:", faiss_pkg)''',
'''# "faiss-gpu" has no pip wheel for Kaggle's python — ensure faiss-cpu is importable.
# (CPU flat inner-product search over 250k x 384 vectors takes milliseconds.)
try:
    import faiss
    print("ok | faiss already available")
except ImportError:
    subprocess.run([sys.executable,"-m","pip","install","-q","faiss-cpu"],check=False)
    print("ok | installed faiss-cpu")''',
"faiss install (gpu wheel doesn't exist on Kaggle)")

# ---- 2a. fit_lgbm returns its OOF array ---------------------------------------
i15 = find_cell("CELL 15 — LIGHTGBM META-MODEL")
repl(i15,
"    return model, t, oof_f1",
"    return model, t, oof_f1, oof_p",
"fit_lgbm returns OOF array")

# ---- 2b. CELL 15: pv from OOF (honest validation view) ------------------------
repl(i15,
'''lgb_ctx, tc, fc = fit_lgbm(Xv, yv, mask_ctx)
lgb_noctx, tn, fn = fit_lgbm(Xv, yv, mask_no)

pv = np.zeros(len(sample))
pv[mask_ctx] = lgbm_predict(Xv.loc[mask_ctx], lgb_ctx)
pv[mask_no] = lgbm_predict(Xv.loc[mask_no], lgb_noctx)''',
'''lgb_ctx, tc, fc, oof_ctx = fit_lgbm(Xv, yv, mask_ctx)
lgb_noctx, tn, fn, oof_no = fit_lgbm(Xv, yv, mask_no)

# honest validation view = OOF probabilities, never the refit model's in-sample fit
pv = np.zeros(len(sample))
pv[mask_ctx] = oof_ctx
pv[mask_no] = oof_no''',
"CELL 15 pv built from OOF")

# ---- 2c. CELL 15.5: same fix + 3. epoch cap -----------------------------------
i155 = find_cell("CELL 15.5 — PSEUDO-LABEL RETRAIN")
repl(i155,
"        m_pseudo, tk_pseudo = train_backbone(bb_key, bb_path, hybrid, sample, seed=SEED + 99)",
'''        _ep = cfg.epochs
        cfg.epochs = min(cfg.epochs, 2)   # round-2 is refinement — 2 epochs protects the 9h budget
        m_pseudo, tk_pseudo = train_backbone(bb_key, bb_path, hybrid, sample, seed=SEED + 99)
        cfg.epochs = _ep''',
"round-2 retrain capped at 2 epochs")

repl(i155,
'''        lgb_ctx, tc, fc = fit_lgbm(Xv, yv, mask_ctx, seed=SEED + 1)
        lgb_noctx, tn, fn = fit_lgbm(Xv, yv, mask_no, seed=SEED + 2)

        pv = np.zeros(len(sample))
        pv[mask_ctx] = lgbm_predict(Xv.loc[mask_ctx], lgb_ctx)
        pv[mask_no] = lgbm_predict(Xv.loc[mask_no], lgb_noctx)''',
'''        lgb_ctx, tc, fc, oof_ctx = fit_lgbm(Xv, yv, mask_ctx, seed=SEED + 1)
        lgb_noctx, tn, fn, oof_no = fit_lgbm(Xv, yv, mask_no, seed=SEED + 2)

        pv = np.zeros(len(sample))
        pv[mask_ctx] = oof_ctx   # OOF -> honest round-2 validation view
        pv[mask_no] = oof_no''',
"CELL 15.5 pv built from OOF")

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

import ast
bad = 0
for j, c in enumerate(cells):
    if c["cell_type"] == "code":
        try:
            ast.parse("".join(c["source"]))
        except SyntaxError as e:
            bad += 1
            print(f"SYNTAX ERROR cell {j}: {e}")
print(f"done: {len(cells)} cells, {bad} syntax errors")
