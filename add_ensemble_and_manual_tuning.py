#!/usr/bin/env python3
"""Updates pipeline.ipynb:
1. Encoder ensemble — adds mDeBERTa-v3 as a second backbone; both are trained
   (or loaded from attached .pt checkpoints) and averaged into one `enc` signal.
   Also fixes the fresh-training bug in CELL 10 (train_backbone returns
   (model, tok), not (preds, tok), and the .pt was loaded before being saved).
2. Manual blend-weight overrides + threshold shift + a weight-sensitivity sweep
   in CELL 15, to counter Powell overfitting the 299-row validation set.
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

def src(i):
    return "".join(cells[i]["source"])

def set_src(i, s):
    cells[i]["source"] = [s]

# ---- 1. CONFIG: two backbones -----------------------------------------------
i = find_cell("CELL 2 — CONFIG")
s = src(i)
old = 'backbones:tuple=(("banglabert_large","csebuetnlp/banglabert_large"),)'
new = ('# Encoder ensemble: each backbone trains (or loads an attached .pt checkpoint)\n'
       '    # and their probabilities are averaged into one "enc" signal before blending.\n'
       '    # Swap "microsoft/mdeberta-v3-base" for "xlm-roberta-large" if time allows\n'
       '    # (xlm-r-large is ~1.7x slower to train, slightly stronger on Bengali).\n'
       '    backbones:tuple=(("banglabert_large","csebuetnlp/banglabert_large"),\n'
       '                     ("mdeberta","microsoft/mdeberta-v3-base"),)')
assert old in s, "backbones line not found"
set_src(i, s.replace(old, new))

# ---- 2. CELL 10: ensemble train/load loop (also fixes fresh-train bug) ------
i = find_cell("CELL 10 — TRAIN")
set_src(i, r'''# ===== CELL 10 — TRAIN: ENCODER ENSEMBLE (BanglaBERT-Large + mDeBERTa) =====
sig_val={}; sig_test={}; keep_for_retr=None
import os, gc, glob, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def find_ckpt(key):
    # a previously-trained checkpoint attached as a dataset skips training entirely
    for pat in (f"/kaggle/input/datasets/bayazidhs/trained-banglabert/{key}.pt",
                f"/kaggle/input/**/{key}.pt"):
        hits = glob.glob(pat, recursive=True)
        if hits: return hits[0]
    return None

first_key = cfg.backbones[0][0]
for bb_key, bb_path in cfg.backbones:
    ckpt = find_ckpt(bb_key)
    if ckpt:
        print(f"[{bb_key}] loading checkpoint {ckpt} — skipping training")
        tk = AutoTokenizer.from_pretrained(bb_path)
        m = AutoModelForSequenceClassification.from_pretrained(
            bb_path, num_labels=2, ignore_mismatched_sizes=True).float().to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    else:
        print(f"[{bb_key}] no checkpoint found — training from scratch")
        m, tk = train_backbone(bb_key, bb_path, train_main, sample)
        torch.save(m.state_dict(), f"/kaggle/working/{bb_key}.pt")  # save BEFORE any reload

    sig_val[bb_key]  = predict_proba(m, tk, sample, cfg.max_len, cfg.batch_size*2)
    sig_test[bb_key] = predict_proba(m, tk, test,   cfg.max_len, cfg.batch_size*2)
    print(f"[{bb_key}] val F1(c0)@0.5 = "
          f"{f1_score(sample['label'],(sig_val[bb_key]>=0.5).astype(int),pos_label=0):.4f}")

    if cfg.use_retrieval and bb_key == first_key:
        keep_for_retr = (m.half().eval(), tk)   # BanglaBERT also scores retrieved passages
    else:
        m = m.cpu(); del m; gc.collect(); torch.cuda.empty_cache()

# --- encoder ensemble: equal-weight average of all backbones -> one "enc" signal ---
enc_keys = [k for k, _ in cfg.backbones if k in sig_val]
sig_val["enc"]  = np.mean([sig_val[k]  for k in enc_keys], axis=0)
sig_test["enc"] = np.mean([sig_test[k] for k in enc_keys], axis=0)
print(f"[enc = avg {enc_keys}] val F1(c0)@0.5 = "
      f"{f1_score(sample['label'],(sig_val['enc']>=0.5).astype(int),pos_label=0):.4f}")
tleft()''')

# ---- 3. CELL 14: stack the averaged "enc" signal ----------------------------
i = find_cell("CELL 14 — RANK-NORMALIZE")
s = src(i)
old = 'X=pd.DataFrame({"bb":sv["bb"],"lex":lex,"retr":retr,"llm":llm})'
new = 'X=pd.DataFrame({"enc":sv["enc"],"lex":lex,"retr":retr,"llm":llm})'
assert old in s, "stackX line not found"
set_src(i, s.replace(old, new))

# ---- 4. CELL 15: manual overrides + sensitivity sweep -----------------------
i = find_cell("CELL 15 — POWELL")
s = src(i)
anchor = 'wc,tc,fc,bc=tune(Xv,yv,~sample["no_ctx"].values); wn,tn,fn,bn=tune(Xv,yv,sample["no_ctx"].values)'
assert anchor in s, "tune anchor not found"
head = s.split(anchor)[0]
set_src(i, head + anchor + r'''

# --- MANUAL OVERRIDES --------------------------------------------------------
# Powell can over-fit the 130/169 validation rows. To hand-tune against the
# public LB: set weights/shift below, re-run this cell + the submission cell.
#   e.g. MANUAL_W_CTX = {"enc":2.0,"lex":1.0,"retr":0.0,"llm":1.5}
MANUAL_W_CTX   = None   # blend weights for has-context rows (None = use Powell's)
MANUAL_W_NOCTX = None   # blend weights for no-context rows  (None = use Powell's)
THR_SHIFT      = 0.0    # + predicts MORE hallucinated (0), - predicts fewer
if MANUAL_W_CTX:   wc = {k: float(MANUAL_W_CTX.get(k, 0.0))   for k in wc}
if MANUAL_W_NOCTX: wn = {k: float(MANUAL_W_NOCTX.get(k, 0.0)) for k in wn}
tc = float(np.clip(tc + THR_SHIFT, 0.05, 0.95))
tn = float(np.clip(tn + THR_SHIFT, 0.05, 0.95))
# ------------------------------------------------------------------------------

print("has_ctx",{k:round(v,1) for k,v in wc.items()},"thr",round(tc,3),"pointF1",round(fc,4),"bootF1",round(bc,4))
print("no_ctx ",{k:round(v,1) for k,v in wn.items()},"thr",round(tn,3),"pointF1",round(fn,4),"bootF1",round(bn,4))
pv=np.where(sample["no_ctx"].values,blend(Xv,wn),blend(Xv,wc)); tv=np.where(sample["no_ctx"].values,tn,tc)
print("OVERALL valF1(c0):",round(f1_score(yv,(pv>=tv).astype(int),pos_label=0),4),
      "| all-0 floor:",round(f1_score(yv,np.zeros(len(yv)),pos_label=0),4))
all0_f1 = f1_score(yv, np.zeros(len(yv)), pos_label=0)
if f1c0(yv, pv, float(np.mean(tv))) < all0_f1 + 0.05:
    print("⚠ Pipeline barely beats all-0 baseline — check signals")

# --- WEIGHT SENSITIVITY (how fragile is Powell's optimum?) --------------------
print("\nweight sensitivity: val F1(c0) when one signal's weight is scaled x0.5 / x1.5")
for reg_name, w0, t0, mask in (("has_ctx", wc, tc, ~sample["no_ctx"].values),
                               ("no_ctx",  wn, tn,  sample["no_ctx"].values)):
    base = f1c0(yv[mask], blend(Xv[mask], w0), t0)
    parts = [f"{reg_name}: base={base:.4f}"]
    for c in w0:
        lo_hi = []
        for sc in (0.5, 1.5):
            w2 = dict(w0); w2[c] = w0[c]*sc
            lo_hi.append(f"{f1c0(yv[mask], blend(Xv[mask], w2), t0):.3f}")
        parts.append(f"{c}[{lo_hi[0]}/{lo_hi[1]}]")
    print("  " + "  ".join(parts))
# big swings on a signal => its Powell weight is unstable; consider a manual override''')

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

import ast
bad = 0
for c in cells:
    if c["cell_type"] == "code":
        try:
            ast.parse("".join(c["source"]))
        except SyntaxError as e:
            bad += 1
            print("SYNTAX ERROR:", e)
print(f"updated {NB}: {len(cells)} cells, {bad} syntax errors")
