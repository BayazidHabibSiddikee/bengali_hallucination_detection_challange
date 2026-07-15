import json

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if 'WEIGHTS_PATH = "/kaggle/input/trained-banglabert/banglabert_large.pt"' in source:
            source = source.replace(
                'WEIGHTS_PATH = "/kaggle/input/trained-banglabert/banglabert_large.pt"',
                'WEIGHTS_PATH = "/kaggle/input/datasets/bayazidhs/trained-banglabert/banglabert_large.pt"'
            )
            
            lines = source.split('\n')
            cell["source"] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Restored original WEIGHTS_PATH")
