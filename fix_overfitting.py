import json

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        
        # 1. Update Weights Paths in Cell 10
        if 'WEIGHTS_PATH = f"/kaggle/input/datasets/bayazidhs/trained-{bb_key}/{bb_key}.pt"' in source:
            new_source = source.replace(
                'WEIGHTS_PATH = f"/kaggle/input/datasets/bayazidhs/trained-{bb_key}/{bb_key}.pt"',
                '# Use the new dataset containing our fixed outputs!\n    if bb_key == "banglabert_large":\n        WEIGHTS_PATH = "/kaggle/input/bengali-pipeline-outputs/banglabert_large_pseudo.pt"\n    elif bb_key == "mdeberta_v3":\n        WEIGHTS_PATH = "/kaggle/input/bengali-pipeline-outputs/mdeberta.pt"\n    else:\n        WEIGHTS_PATH = f"/kaggle/input/datasets/bayazidhs/trained-{bb_key}/{bb_key}.pt"'
            )
            
            # Remove the old fallback logic to avoid conflicts
            new_source = new_source.replace(
                'if bb_key == "banglabert_large" and not os.path.exists(WEIGHTS_PATH):\n        WEIGHTS_PATH = "/kaggle/input/datasets/bayazidhs/trained-banglabert/banglabert_large.pt"',
                ''
            )
            
            lines = new_source.split('\n')
            cell["source"] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

        # 2. Fix LightGBM Overfitting in Cell 15
        if 'num_leaves=15' in source and 'min_data_in_leaf=3' in source:
            new_source = source.replace('num_leaves=15', 'num_leaves=3')
            new_source = new_source.replace('min_data_in_leaf=3', 'min_data_in_leaf=15')
            new_source = new_source.replace('num_boost_round=150', 'num_boost_round=35')
            new_source = new_source.replace('learning_rate=0.05', 'learning_rate=0.01')
            
            lines = new_source.split('\n')
            cell["source"] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("✅ Updated pipeline.ipynb to prevent LightGBM overfitting and point to the new dataset paths!")
