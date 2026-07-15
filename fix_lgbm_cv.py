import json

def fix_lgbm_cell(filename):
    with open(filename) as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell.get('source', []))
            if 'def fit_lgbm(' in src and 'def tune_threshold(' in src:
                print(f'Patching {filename} Cell {i}')
                # Write the new cell content
                new_src = """# ===== CELL 15 — LIGHTGBM META-MODEL STACKING (replaces Powell blender) =====
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

def f1c0(yy, p, t):
    return f1_score(yy, (p >= t).astype(int), pos_label=0)

FEAT_COLS = [c for c in Xv.columns if c != "no_ctx"]

def tune_threshold(p, y, n_boot=cfg.n_boot, seed=SEED):
    m = len(y)
    rng = np.random.RandomState(seed)
    grid = np.quantile(p, np.linspace(0.05, 0.95, 60))
    picks = []
    for _ in range(n_boot):
        b = rng.randint(0, m, m)
        pb, yb = p[b], y[b]
        pred0 = (pb[:, None] < grid[None, :]).astype(np.float32)
        tp = ((yb == 0)[:, None] * pred0).sum(0)
        f1 = 2 * tp / np.maximum(pred0.sum(0) + (yb == 0).sum(), 1e-9)
        picks.append(grid[int(f1.argmax())])
    return float(np.median(picks))

def fit_lgbm(X, y, mask, seed=SEED):
    Xr = X.loc[mask, FEAT_COLS].reset_index(drop=True)
    yr = y[mask]
    
    params = dict(
        objective="binary", metric="binary_logloss", verbosity=-1, seed=seed,
        learning_rate=0.01, num_leaves=3, min_data_in_leaf=15,
        feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
    )
    
    # 5-Fold OOF Predictions for Threshold Tuning
    oof_p = np.zeros(len(yr))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    for trn_idx, val_idx in skf.split(Xr, yr):
        X_trn, y_trn = Xr.iloc[trn_idx], yr[trn_idx]
        X_val = Xr.iloc[val_idx]
        
        m_cv = lgb.train(params, lgb.Dataset(X_trn, label=y_trn), num_boost_round=35)
        oof_p[val_idx] = m_cv.predict(X_val)
        
    t = tune_threshold(oof_p, yr)
    oof_f1 = f1c0(yr, oof_p, t)
    
    # Final Model trained on 100% of data
    model = lgb.train(params, lgb.Dataset(Xr, label=yr), num_boost_round=35)
    return model, t, oof_f1

def lgbm_predict(X, model):
    return model.predict(X[FEAT_COLS])

mask_ctx = ~sample["no_ctx"].values
mask_no = sample["no_ctx"].values
lgb_ctx, tc, fc = fit_lgbm(Xv, yv, mask_ctx)
lgb_noctx, tn, fn = fit_lgbm(Xv, yv, mask_no)

pv = np.zeros(len(sample))
pv[mask_ctx] = lgbm_predict(Xv.loc[mask_ctx], lgb_ctx)
pv[mask_no] = lgbm_predict(Xv.loc[mask_no], lgb_noctx)
tv = np.where(sample["no_ctx"].values, tn, tc)

pt = np.zeros(len(test))
pt[~test["no_ctx"].values] = lgbm_predict(Xt.loc[~test["no_ctx"].values], lgb_ctx)
pt[test["no_ctx"].values] = lgbm_predict(Xt.loc[test["no_ctx"].values], lgb_noctx)
tt = np.where(test["no_ctx"].values, tn, tc)

THR_SHIFT = 0.0
tc = float(np.clip(tc + THR_SHIFT, 0.05, 0.95))
tn = float(np.clip(tn + THR_SHIFT, 0.05, 0.95))
tv = np.where(sample["no_ctx"].values, tn, tc)
tt = np.where(test["no_ctx"].values, tn, tc)

print("LGBM has_ctx thr", round(tc, 3), "OOF pointF1", round(fc, 4))
print("LGBM no_ctx  thr", round(tn, 3), "OOF pointF1", round(fn, 4))
print(
    "OVERALL OOF F1(c0):",
    round(f1_score(yv, (pv >= tv).astype(int), pos_label=0), 4),
    "| all-0 floor:",
    round(f1_score(yv, np.zeros(len(yv)), pos_label=0), 4),
)

imp_ctx = pd.Series(lgb_ctx.feature_importance(), index=FEAT_COLS).sort_values(ascending=False)
imp_no = pd.Series(lgb_noctx.feature_importance(), index=FEAT_COLS).sort_values(ascending=False)
print("top features has_ctx:", {k: int(v) for k, v in imp_ctx.head(5).items()})
print("top features no_ctx:", {k: int(v) for k, v in imp_no.head(5).items()})
"""
                
                # Split lines keeping newlines
                new_src_lines = [line + '\n' for line in new_src.split('\n')]
                new_src_lines[-1] = new_src_lines[-1].rstrip('\n')
                nb['cells'][i]['source'] = new_src_lines

    with open(filename, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

fix_lgbm_cell('pipeline.ipynb')
fix_lgbm_cell('bengali-hallu.ipynb')
