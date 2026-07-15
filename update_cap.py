import json

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "max_train_rows" in source:
            source = source.replace("max_train_rows:int=110000", "max_train_rows:int=15000")
            cell["source"] = [source]
        
        if "qa_items[:40000]" in source:
            source = source.replace("qa_items[:40000]", "qa_items[:10000]")
            cell["source"] = [source]

with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

