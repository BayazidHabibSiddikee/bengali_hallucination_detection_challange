import json

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        
        # Fix IndicXNLI trust_remote_code
        if "load_dataset(cfg.hf_ixnli," in source:
            source = source.replace('load_dataset(cfg.hf_ixnli,"bn",split="train",token=HF_TOKEN)', 'load_dataset(cfg.hf_ixnli,"bn",split="train",token=HF_TOKEN,trust_remote_code=True)')
            cell["source"] = [source]
            
with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

