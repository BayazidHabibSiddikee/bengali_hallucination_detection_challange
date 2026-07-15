import json
import os

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

os.makedirs("scratch", exist_ok=True)
cell_idx = 0
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        with open(f"scratch/cell_{cell_idx}.py", "w", encoding="utf-8") as f:
            f.write(source)
    cell_idx += 1
