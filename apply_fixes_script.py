import json
import re

def fix_pipeline():
    with open('pipeline.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # FIX 1: Cell 14 (Regime Flags)
            if "# ===== CELL 14 — RANK-NORMALIZE SIGNALS" in source:
                # Replace the old regime flags section
                old_regime = """    # --- regime flags (Task/Regime Router -> meta-model) ---
    pr = df["prompt_bn"].astype(str); rs = df["response_bn"].astype(str)
    cx = df["ctx_clean"].astype(str)
    X["is_math"] = [int(bool(numset(p)) and bool(numset(r))) for p, r in zip(pr, rs)]
    X["is_translation"] = pr.str.contains(
        "অনুবাদ|translate|ইংরেজিতে|সারাংশ|সংক্ষেপে|summar", regex=True, case=False).astype(int).values
    X["is_mcq"] = (pr.str.contains(r"ক\)", regex=True)
                   & pr.str.contains(r"খ\)", regex=True)).astype(int).values
    def _numsup(p, r, c):
        # fraction of response numbers that also appear in prompt+context
        # (Bengali numerals normalized); -1 = response has no numbers
        nr = numset(r)
        return -1.0 if not nr else len(nr & (numset(p) | numset(c))) / len(nr)
    X["number_support"] = [_numsup(p, r, c) for p, r, c in zip(pr, rs, cx)]"""
                
                new_regime = """    # --- regime flags (Task/Regime Router -> meta-model) ---
    TRANSLATE_PAT = re.compile(r'অনুবাদ|translate|ইংরেজিতে|বাংলায়|রূপান্তর', re.I)
    SUMMARY_PAT   = re.compile(r'সারাংশ|সংক্ষেপে|summar', re.I)
    pr = df["prompt_bn"].astype(str)
    rs = df["response_bn"].astype(str)
    cx = df["ctx_clean"].astype(str)
    
    X["regime_context"]     = (~df["no_ctx"].values).astype(float)
    X["regime_factual"]     = (df["no_ctx"].values & ~pr.str.contains(TRANSLATE_PAT) & ~pr.str.contains(SUMMARY_PAT)).astype(float)
    X["regime_translation"] = pr.str.contains(TRANSLATE_PAT, regex=True, na=False).astype(float)
    X["regime_summary"]     = pr.str.contains(SUMMARY_PAT, regex=True, na=False).astype(float)
    
    X["is_math"] = [int(bool(numset(p)) and bool(numset(r))) for p, r in zip(pr, rs)]
    X["is_translation"] = pr.str.contains("অনুবাদ|translate|ইংরেজিতে|সারাংশ|সংক্ষেপে|summar", regex=True, case=False).astype(int).values
    X["is_mcq"] = (pr.str.contains(r"ক\)", regex=True) & pr.str.contains(r"খ\)", regex=True)).astype(int).values
    def _numsup(p, r, c):
        nr = numset(r)
        return -1.0 if not nr else len(nr & (numset(p) | numset(c))) / len(nr)
    X["number_support"] = [_numsup(p, r, c) for p, r, c in zip(pr, rs, cx)]"""
                
                source = source.replace(old_regime, new_regime)
                # Split back into lines
                cell['source'] = [line + '\n' for line in source.split('\n')]
                # Fix trailing newline
                if cell['source'] and cell['source'][-1].endswith('\n\n'):
                    cell['source'][-1] = cell['source'][-1][:-1]
                
            
            # FIX 2: Cell 15 (Platt Calibration)
            elif "# ===== CELL 15 — LIGHTGBM META-MODEL STACKING" in source:
                # Add imports
                if "from sklearn.calibration import CalibratedClassifierCV" not in source:
                    source = source.replace("import lightgbm as lgb", "import lightgbm as lgb\nfrom sklearn.calibration import CalibratedClassifierCV\nfrom sklearn.linear_model import LogisticRegression")
                
                # We need to replace the bottom part of cell 15
                old_bot = """pv = np.zeros(len(sample))
pv[mask_ctx] = oof_ctx
pv[mask_no] = oof_no
tv = np.where(sample["no_ctx"].values, tn, tc)

pt = np.zeros(len(test))
pt[~test["no_ctx"].values] = lgbm_predict(Xt.loc[~test["no_ctx"].values], lgb_ctx)
pt[test["no_ctx"].values] = lgbm_predict(Xt.loc[test["no_ctx"].values], lgb_noctx)
tt = np.where(test["no_ctx"].values, tn, tc)

THR_SHIFT = 0.08
tc = float(np.clip(tc + THR_SHIFT, 0.05, 0.95))
tn = float(np.clip(tn + THR_SHIFT, 0.05, 0.95))
tv = np.where(sample["no_ctx"].values, tn, tc)
tt = np.where(test["no_ctx"].values, tn, tc)"""

                new_bot = """def calibrate_probs(oof_p, y):
    cal = LogisticRegression(C=1.0, max_iter=1000)
    cal.fit(oof_p.reshape(-1,1), y)
    return cal

def apply_calibration(cal, p):
    return cal.predict_proba(p.reshape(-1,1))[:,1]

cal_ctx   = calibrate_probs(oof_ctx, yv[mask_ctx])
cal_noctx = calibrate_probs(oof_no,  yv[mask_no])

oof_ctx_cal = apply_calibration(cal_ctx,   oof_ctx)
oof_no_cal  = apply_calibration(cal_noctx, oof_no)

pv = np.zeros(len(sample))
pv[mask_ctx] = oof_ctx_cal
pv[mask_no]  = oof_no_cal

pt = np.zeros(len(test))
pt_ctx_raw = lgbm_predict(Xt.loc[~test["no_ctx"].values], lgb_ctx)
pt_no_raw  = lgbm_predict(Xt.loc[test["no_ctx"].values], lgb_noctx)
pt[~test["no_ctx"].values] = apply_calibration(cal_ctx,   pt_ctx_raw)
pt[test["no_ctx"].values]  = apply_calibration(cal_noctx, pt_no_raw)

tc = tune_threshold(oof_ctx_cal, yv[mask_ctx])
tn = tune_threshold(oof_no_cal,  yv[mask_no])

tv = np.where(sample["no_ctx"].values, tn, tc)
tt = np.where(test["no_ctx"].values, tn, tc)"""
                
                source = source.replace(old_bot, new_bot)
                cell['source'] = [line + '\n' for line in source.split('\n')]
                if cell['source'] and cell['source'][-1].endswith('\n\n'):
                    cell['source'][-1] = cell['source'][-1][:-1]
                cell_15_idx = idx

    # FIX 3: Cell 15.5 (Pseudo Label Retrain)
    # Check if Cell 15.5 already exists to avoid duplicates
    has_pseudo = False
    for cell in nb['cells']:
        if "# ===== CELL 15.5 — PSEUDO LABEL RETRAIN" in "".join(cell.get('source', [])):
            has_pseudo = True
            break
            
    if not has_pseudo:
        pseudo_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ===== CELL 15.5 — PSEUDO LABEL RETRAIN =====\n",
                "if cfg.pseudo_label_n > 0:\n",
                "    conf_mask = (pt < (tt - 0.25)) | (pt > (tt + 0.25))\n",
                "    pseudo_df = test[conf_mask].copy()\n",
                "    pseudo_df[\"label\"] = (pt[conf_mask] >= tt[conf_mask]).astype(int)\n",
                "    pseudo_df = pseudo_df.nlargest(\n",
                "        min(cfg.pseudo_label_n, len(pseudo_df)),\n",
                "        key=lambda x: abs(pt[conf_mask] - tt[conf_mask])\n",
                "    )\n",
                "    pseudo_df[\"premise\"]  = pseudo_df[\"premise\"]\n",
                "    pseudo_df[\"response\"] = pseudo_df[\"response_bn\"].astype(str)\n",
                "    pseudo_df[\"src\"]      = \"test_set\"\n",
                "    pseudo_df[\"mode\"]     = \"pseudo\"\n",
                "    pseudo_df[[\"premise\",\"response\",\"label\",\"src\",\"mode\"]].to_csv(\n",
                "        \"/kaggle/working/pseudo_labels.csv\", index=False)\n",
                "    print(f\"Saved {len(pseudo_df)} pseudo labels\")"
            ]
        }
        nb['cells'].insert(cell_15_idx + 1, pseudo_cell)

    with open('pipeline.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    fix_pipeline()
