import json
import re

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    source_list = cell.get("source", [])
    source = "".join(source_list)
    
    # 1. train_backbone CPU clear
    if "def train_backbone" in source:
        if "del opt, scaler" not in source:
            old = "    return model,tok"
            new = "    model = model.cpu()\n    del opt, scaler, ld, crit\n    import gc; gc.collect(); torch.cuda.empty_cache()\n    return model,tok"
            source = source.replace(old, new)
            
    # 2. Cell 13 reduce tokens
    if "def run_llm_judge(df):" in source:
        source = source.replace('enc["input_ids"][:, -768:]', 'enc["input_ids"][:, -512:]')
        source = source.replace('enc["attention_mask"][:, -768:]', 'enc["attention_mask"][:, -512:]')
        # Also ensure max_mem is using 6GiB and 10GiB as we told them, or cfg if they want
        # Just leave max_mem as it is because we already helped them set it
        
    # 3. Batch size to 8 in CFG
    if "class CFG:" in source:
        source = re.sub(r'batch_size\s*=\s*\d+', 'batch_size = 8', source)

    # 4. In build_retr_signal, use weighted average (mean) instead of max
    if "def build_retr_signal" in source:
        # Currently the code has: pp.reshape(len(idx),cfg.retr_topk).max(1)
        old_max = "pp.reshape(len(idx),cfg.retr_topk).max(1)"
        new_mean = "pp.reshape(len(idx),cfg.retr_topk).mean(1)"
        source = source.replace(old_max, new_mean)
        
    # Put lines back
    # Simple splitlines(True) might not exactly match the original list format but Jupyter loads it fine.
    # To be safe, we split by '\n' and append '\n' except for the last line.
    lines = source.split('\n')
    new_source_list = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
    cell["source"] = new_source_list
    
with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    
print("Improvements applied.")
