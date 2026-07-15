import json
import re

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    
    # Change 1: add model.cpu() + delete inside train_backbone()
    if "def train_backbone" in source:
        if "del opt, scaler" not in source:
            old = "    return model,tok"
            new = "    model = model.cpu()\n    del opt, scaler, ld; gc.collect(); torch.cuda.empty_cache()\n    return model,tok"
            source = source.replace(old, new)
            
    # Change 2: dynamic memory budget in nuclear_clear()
    # Wait, the user mentions nuclear_clear(). Let's see if it exists.
    
    # Change 3: Cell 13 - use cfg.max_mem_llm + reduce to 512 tokens + fallback chain
    if "def run_llm_judge(df):" in source:
        # max_mem
        old = 'max_mem = {0: "5GiB", 1: "12GiB", "cpu": "25GiB"}'
        if old in source or 'max_mem = {0: "6GiB", 1: "10GiB", "cpu": "25GiB"}' in source:
            source = re.sub(r'max_mem = \{.*?\}', 'max_mem = getattr(cfg, "max_mem_llm", {0: "12GiB", 1: "12GiB", "cpu": "30GiB"})', source)
        
        # reduce to 512 tokens
        old_tokens = 'enc["input_ids"][:, -768:]'
        new_tokens = 'enc["input_ids"][:, -512:]'
        source = source.replace(old_tokens, new_tokens)
        
        old_attn = 'enc["attention_mask"][:, -768:]'
        new_attn = 'enc["attention_mask"][:, -512:]'
        source = source.replace(old_attn, new_attn)
        
        # fallback chain? Let's check how to implement fallback chain or skip if unsure.
        
    # Change 6: Cell 10 - weighted mean instead of max for retrieval
    if "def build_retr_signal" in source:
        old_max = "np.max([score(c) for c in ctx_list])"
        new_mean = "np.average([score(c) for c in ctx_list], weights=range(len(ctx_list), 0, -1)) if ctx_list else np.nan"
        source = source.replace(old_max, new_mean)
        
    # Change 7: Cell 1 - batch_size=8, add llm_input_len=512
    if "class CFG:" in source:
        source = re.sub(r'batch_size\s*=\s*\d+', 'batch_size = 8', source)
        if "llm_input_len" not in source:
            source = source.replace('batch_size = 8', 'batch_size = 8\n    llm_input_len = 512')

    # Update cell source
    cell["source"] = [source]
    
with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    
print("Basic fixes applied to pipeline.ipynb")
