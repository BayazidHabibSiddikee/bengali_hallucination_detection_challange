import json

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    
    if "m, tk = train_backbone(bb_key" in source and "train_main" in source:
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
    m = m.cpu()
else:
    print("⚠ Weights not found — running full training...")
    m, tk = train_backbone(bb_key, bb_path, train_main, sample)

m = m.to(DEVICE)
sig_val["bb"] = predict_proba(m, tk, sample, cfg.max_len, cfg.batch_size*2)
sig_test["bb"] = predict_proba(m, tk, test, cfg.max_len, cfg.batch_size*2)
m = m.cpu()

torch.save(m.state_dict(), f"/kaggle/working/{bb_key}.pt")

if cfg.use_retrieval:
    keep_for_retr = (m.half().eval(), tk)
else:
    del m; import gc; gc.collect(); torch.cuda.empty_cache()

tleft()
"""
        # Overwrite source
        lines = new_source.split('\n')
        cell["source"] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
        
with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Applied Cell 10 fallback!")
