import json
import re

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    source_list = cell.get("source", [])
    source = "".join(source_list)
    
    if "for p in" in source and "transformers>=4.44" in source:
        # We find the list and append lightgbm
        # The list currently is ["transformers>=4.44","sentencepiece","accelerate>=0.30","bitsandbytes","datasets==2.19.0","tqdm"]
        if '"lightgbm"' not in source and "'lightgbm'" not in source:
            source = source.replace('"tqdm"]', '"tqdm","lightgbm"]')
            source = source.replace("'tqdm']", "'tqdm','lightgbm']")
            
            lines = source.split('\n')
            new_source_list = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
            cell["source"] = new_source_list
            break
            
with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("LightGBM added to install cell")
