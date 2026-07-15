import json

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 11 for Overlapping Chunks
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        
        # Overlapping chunks injection in Cell 11
        if 'for p in wiki_passages:' in source and 'for i in range(0,len(p),400):' in source:
            new_source = source.replace(
                'for i in range(0,len(p),400):\n            c=p[i:i+500]',
                'chunk_size = 500\n        overlap = 150\n        step = chunk_size - overlap\n        for i in range(0, max(1, len(p) - overlap), step):\n            c = p[i:i+chunk_size]'
            )
            lines = new_source.split('\n')
            cell["source"] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

# 2. Add Pseudo-Labeling Cell before Cell 18
pseudo_code = """# ===== CELL 17.5 — EXPORT PSEUDO-LABELS FOR NEXT RUN =====
import pandas as pd
if 'test' in globals() and 'pt' in globals():
    print("Generating Pseudo-Labels from highly confident test predictions...")
    test_pseudo = test.copy()
    test_pseudo['prob'] = pt  # 'pt' comes from the Powell Optimizer final blend
    
    # Filter for extreme confidence (>95% sure it's faithful, <5% sure it's hallucination)
    conf_faithful = test_pseudo[test_pseudo['prob'] >= 0.95].copy()
    conf_hallu = test_pseudo[test_pseudo['prob'] <= 0.05].copy()
    
    conf_faithful['label'] = 1
    conf_hallu['label'] = 0
    
    pseudo_df = pd.concat([conf_faithful, conf_hallu])[['premise', 'response', 'label']]
    pseudo_df['mode'] = 'pseudo'
    pseudo_df['src'] = 'test_set'
    
    pseudo_df.to_csv("/kaggle/working/pseudo_labels.csv", index=False)
    print(f"✅ Saved {len(pseudo_df)} highly confident test predictions to pseudo_labels.csv!")
    print("💡 TIP: Download this file, upload it as a Kaggle dataset, and in your next run, append it to train_all in Cell 8 to boost accuracy!")
"""

new_pseudo_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + '\n' for line in pseudo_code.split('\n')[:-1]] + [pseudo_code.split('\n')[-1]]
}

# Insert before the last cell (which is our graph cell)
nb["cells"].insert(-1, new_pseudo_cell)

with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Added overlapping chunks and pseudo-labeling exporter to pipeline.ipynb")
