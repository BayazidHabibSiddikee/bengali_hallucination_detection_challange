import json
import re

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        
        # Update Weights Paths in Cell 10 to use the user's specific dataset path
        if 'WEIGHTS_PATH = "/kaggle/input/bengali-pipeline-outputs/banglabert_large_pseudo.pt"' in source:
            new_source = source.replace(
                'WEIGHTS_PATH = "/kaggle/input/bengali-pipeline-outputs/banglabert_large_pseudo.pt"',
                'WEIGHTS_PATH = "/kaggle/input/bengali-trained-mdeberta/banglabert_large_pseudo.pt"\n        if not os.path.exists(WEIGHTS_PATH):\n            WEIGHTS_PATH = "/kaggle/input/datasets/bayazidhs/bengali-trained-mdeberta/banglabert_large_pseudo.pt"'
            )
            new_source = new_source.replace(
                'WEIGHTS_PATH = "/kaggle/input/bengali-pipeline-outputs/mdeberta.pt"',
                'WEIGHTS_PATH = "/kaggle/input/bengali-trained-mdeberta/mdeberta.pt"\n        if not os.path.exists(WEIGHTS_PATH):\n            WEIGHTS_PATH = "/kaggle/input/datasets/bayazidhs/bengali-trained-mdeberta/mdeberta.pt"'
            )
            
            lines = new_source.split('\n')
            cell["source"] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("✅ Updated pipeline.ipynb to use the exact bengali-trained-mdeberta paths!")
