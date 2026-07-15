import json

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        
        # 1. Update CFG to include xlm-roberta-large
        if 'backbones:tuple=(("banglabert_large","csebuetnlp/banglabert_large"),)' in source:
            new_source = source.replace(
                'backbones:tuple=(("banglabert_large","csebuetnlp/banglabert_large"),)',
                'backbones:tuple=(("banglabert_large","csebuetnlp/banglabert_large"), ("xlm_roberta","xlm-roberta-large"))'
            )
            lines = new_source.split('\n')
            cell["source"] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

        # 2. Rewrite Cell 10 to loop over multiple backbones and average them
        if 'bb_key = list(dict(cfg.backbones).keys())[0]' in source:
            new_cell_10 = """# ===== CELL 10 — TRAIN: MULTI-MODEL ENSEMBLE =====
sig_val={}; sig_test={}; keep_for_retr=None
import os, torch, gc
from transformers import AutoModelForSequenceClassification, AutoTokenizer

bb_val_preds = []
bb_test_preds = []

# Loop through ALL models defined in cfg.backbones
for bb_key, bb_path in cfg.backbones:
    print(f"\\n{'='*40}\\nProcessing Backbone: {bb_key}\\n{'='*40}")
    
    # Check if we have pre-trained weights uploaded for this specific model
    WEIGHTS_PATH = f"/kaggle/input/datasets/bayazidhs/trained-{bb_key}/{bb_key}.pt"
    # Fallback to the known banglabert path just in case
    if bb_key == "banglabert_large" and not os.path.exists(WEIGHTS_PATH):
        WEIGHTS_PATH = "/kaggle/input/datasets/bayazidhs/trained-banglabert/banglabert_large.pt"

    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading pretrained weights for {bb_key}, skipping training...")
        m = AutoModelForSequenceClassification.from_pretrained(bb_path, num_labels=2, ignore_mismatched_sizes=True).to(DEVICE)
        m.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        tk = AutoTokenizer.from_pretrained(bb_path)
        
        v_pred = predict_proba(m, tk, sample, cfg.max_len, cfg.batch_size*2)
    else:
        print(f"⚠ Weights not found for {bb_key} — running full training (this will take time)...")
        v_pred, tk = train_backbone(bb_key, bb_path, train_main, sample)
        
        # Load best model fresh for test predictions
        m = AutoModelForSequenceClassification.from_pretrained(bb_path, num_labels=2, ignore_mismatched_sizes=True).to(DEVICE)
        m.load_state_dict(torch.load(f"/kaggle/working/{bb_key}.pt", map_location=DEVICE))

    # Get test predictions for this model
    t_pred = predict_proba(m, tk, test, cfg.max_len, cfg.batch_size*2)
    
    bb_val_preds.append(v_pred)
    bb_test_preds.append(t_pred)
    
    # We only need to keep ONE model in memory for the Retrieval TF-IDF fallback step
    if cfg.use_retrieval and keep_for_retr is None:
        keep_for_retr = (m.half().to(DEVICE).eval(), tk)
        print(f"Keeping {bb_key} in memory for TF-IDF Retrieval step.")
    else:
        m = m.cpu()
        del m; gc.collect(); torch.cuda.empty_cache()

# Average all the backbone predictions together to create the Ultimate Ensemble!
import numpy as np
sig_val["bb"] = np.mean(bb_val_preds, axis=0)
sig_test["bb"] = np.mean(bb_test_preds, axis=0)
print(f"\\n✅ Successfully ensembled {len(bb_val_preds)} backbone models!")

tleft()
"""
            lines = new_cell_10.split('\n')
            cell["source"] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated pipeline.ipynb with Multi-Model Ensemble logic.")
