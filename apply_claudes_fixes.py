import json
import re

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    
    # Fix 1: IndicXNLI cap (40000 -> 15000)
    if "def load_indicxnli():" in source:
        source = source.replace("min(40000,", "min(15000,")
        
    # Fix 2 & 4: train_backbone return & model.half().to(DEVICE)
    if "def train_backbone" in source:
        # We previously changed it to:
        # model = model.cpu()
        # del opt, scaler, ld, crit
        # import gc; gc.collect(); torch.cuda.empty_cache()
        # return model,tok
        
        # We need to change it to:
        # val_preds = predict_proba(model, tok, val, cfg.max_len, cfg.batch_size*2)
        # model = model.cpu()
        # del model, opt, scaler, ld, crit
        # import gc; gc.collect(); torch.cuda.empty_cache()
        # return val_preds, tok
        
        # Let's rebuild the end of the function carefully
        old_end = """        f=f1_score(val["label"],(predict_proba(model,tok,val,cfg.max_len,cfg.batch_size*2)>=0.5).astype(int),pos_label=0)
        if not quiet: print(f"  {name} s{seed} ep{ep+1}: valF1(c0)={f:.4f}")
    model = model.cpu()
    del opt, scaler, ld, crit
    import gc; gc.collect(); torch.cuda.empty_cache()
    return model,tok"""
        
        new_end = """        f=f1_score(val["label"],(predict_proba(model,tok,val,cfg.max_len,cfg.batch_size*2)>=0.5).astype(int),pos_label=0)
        if not quiet: print(f"  {name} s{seed} ep{ep+1}: valF1(c0)={f:.4f}")
    val_preds = predict_proba(model, tok, val, cfg.max_len, cfg.batch_size*2)
    model = model.cpu()
    del model, opt, scaler, ld, crit
    import gc; gc.collect(); torch.cuda.empty_cache()
    return val_preds, tok"""
        source = source.replace(old_end, new_end)
        
    # Fix 3: Cell 11 loading and predicting with fresh model
    if "sig_val={}; sig_test={}; keep_for_retr=None" in source and "bb_key = list(dict(cfg.backbones).keys())[0]" in source:
        # This is the fallback cell we wrote. Let's rewrite it.
        new_source = """# ===== CELL 10 — TRAIN: BanglaBERT-Large ONLY =====
sig_val={}; sig_test={}; keep_for_retr=None

bb_key = list(dict(cfg.backbones).keys())[0]
bb_path = dict(cfg.backbones)[bb_key]

WEIGHTS_PATH = "/kaggle/input/trained-banglabert/banglabert_large.pt"
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if os.path.exists(WEIGHTS_PATH):
    print("Loading pretrained weights, skipping 3-hour training...")
    m = AutoModelForSequenceClassification.from_pretrained(bb_path, num_labels=2).to(DEVICE)
    m.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    tk = AutoTokenizer.from_pretrained(bb_path)
    sig_val["bb"] = predict_proba(m, tk, sample, cfg.max_len, cfg.batch_size*2)
else:
    print("⚠ Weights not found — running full training...")
    val_preds, tk = train_backbone(bb_key, bb_path, train_main, sample)
    sig_val["bb"] = val_preds
    # Load model fresh for test predictions
    m = AutoModelForSequenceClassification.from_pretrained(bb_path, num_labels=2).to(DEVICE)
    m.load_state_dict(torch.load(f"/kaggle/working/{bb_key}.pt", map_location=DEVICE))

sig_test["bb"] = predict_proba(m, tk, test, cfg.max_len, cfg.batch_size*2)

if cfg.use_retrieval:
    keep_for_retr = (m.half().to(DEVICE).eval(), tk)
else:
    m = m.cpu()
    del m; import gc; gc.collect(); torch.cuda.empty_cache()

tleft()
"""
        source = new_source
        
    # Fix 5: TigerLLM Judge input length hardcoded & dev variable
    if "LLM_FALLBACKS =" in source:
        # We need to make sure 768 is replaced by cfg.llm_input_len and dev is set after loop
        # My previous script might have failed if it was replaced with LLM_INPUT_LEN or something
        source = re.sub(r'\[:, -768:\]', '[:, -cfg.llm_input_len:]', source)
        source = re.sub(r'\[:, -512:\]', '[:, -cfg.llm_input_len:]', source)
        
        # Also ensure dev = torch.device("cuda:0") is removed if it's there
        source = source.replace('dev = torch.device("cuda:0")', '')
        # dev is already set via dev = next(llm.parameters()).device in my previous code block.
        
    lines = source.split('\n')
    cell["source"] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Applied Claude's requested feedback")
