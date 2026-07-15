import json
import re

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Insert cell 12.5 before cell 13
new_cells = []
for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        new_cells.append(cell)
        continue
    source = "".join(cell.get("source", []))
    
    # 1. Insert Cell 12.5 (nuclear_clear) right after Cell 12 (LEX/NUM)
    if "def lexnum(df):" in source:
        new_cells.append(cell)
        
        # Build nuclear clear cell
        nuclear_code = """# ===== CELL 12.5 — NUCLEAR CLEAR & MEMORY BUDGET =====
import gc, torch, ctypes

def nuclear_clear():
    suspects = ['m','model','tk','tok','tokenizer','keep_for_retr',
                'opt','sch','scaler','ld','crit','backbone']
    for name in suspects:
        if name in globals(): del globals()[name]
    
    for _ in range(3): gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except: pass
    
    if torch.cuda.is_available():
        total_free = 0
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            total_free += free
            alloc = torch.cuda.memory_allocated(i)/1e9
            print(f"GPU{i}: {free/1e9:.1f}GB free / {total/1e9:.1f}GB | alloc={alloc:.1f}GB")
        
        free0 = torch.cuda.mem_get_info(0)[0]
        free1 = torch.cuda.mem_get_info(1)[0]
        cfg.max_mem_llm = {
            0: f"{max(1, int(free0/1e9*0.75))}GiB",
            1: f"{max(1, int(free1/1e9*0.85))}GiB",
            "cpu": "40GiB"
        }
        print(f"LLM budget: {cfg.max_mem_llm}")
        
        if total_free < 8e9:
            print("⚠ Less than 8GB free total — LLM may OOM")
        else:
            print("✅ GPU clear — safe to load LLM")

nuclear_clear()
"""
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + '\n' for line in nuclear_code.split('\n')[:-1]] + [nuclear_code.split('\n')[-1]]
        }
        new_cells.append(new_cell)
        continue
        
    # 2. Update run_llm_judge (Cell 13)
    if "def run_llm_judge(df):" in source:
        # We need to replace the LLM loading part with fallback and use cfg.max_mem_llm
        old_loading = """    path = resolve_model("TigerLLM-9B-it", cfg.llm_id)
    tk = AutoTokenizer.from_pretrained(path)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    
    # Force it to perfectly balance 12GB max per GPU to prevent crashing
    max_mem = {0: "5GiB", 1: "12GiB", "cpu": "25GiB"}
    llm = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, device_map="balanced", max_memory=max_mem).eval()
    dev = torch.device("cuda:0")"""
        
        # It might also look like the original if it wasn't replaced cleanly. Let's do a robust regex or block replace.
        # Just find from `path = resolve_model` up to `dev = ...`
        start_idx = source.find('    path = resolve_model')
        if start_idx != -1:
            end_idx = source.find('    def digit_ids', start_idx)
            if end_idx != -1:
                new_loading = """    max_mem = cfg.max_mem_llm
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    LLM_FALLBACKS = [ ("TigerLLM-9B-it", cfg.llm_id), ("Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"), ("Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct") ]
    llm = None
    for model_name, model_id in LLM_FALLBACKS:
        try:
            path = resolve_model(model_name, model_id)
            tk = AutoTokenizer.from_pretrained(path)
            llm = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, device_map="balanced", max_memory=max_mem).eval()
            print(f"✅ Loaded: {model_name}")
            break
        except RuntimeError as e:
            if "memory" in str(e).lower():
                print(f"⚠ OOM on {model_name}, trying fallback...")
                import gc; gc.collect(); torch.cuda.empty_cache()
            else:
                raise
    if llm is None:
        print("❌ All LLMs failed — skipping judge")
        return np.full(len(df), np.nan)
    dev = next(llm.parameters()).device
"""
                source = source[:start_idx] + new_loading + source[end_idx:]
                
        # Update input tokens if not 512
        source = source.replace('enc["input_ids"][:, -512:]', 'enc["input_ids"][:, -cfg.llm_input_len:]')
        source = source.replace('enc["attention_mask"][:, -512:]', 'enc["attention_mask"][:, -cfg.llm_input_len:]')
        source = source.replace('enc["input_ids"][:, -768:]', 'enc["input_ids"][:, -cfg.llm_input_len:]')
        source = source.replace('enc["attention_mask"][:, -768:]', 'enc["attention_mask"][:, -cfg.llm_input_len:]')
        
    # 3. Update tune (Cell 15)
    if "def tune(Xv" in source:
        source = source.replace('n_boot=60', 'n_boot=cfg.n_boot')
        
        # Add all-0 guard
        if "OVERALL valF1" in source and "all0_f1" not in source:
            source += """
all0_f1 = f1_score(yv, np.zeros(len(yv)), pos_label=0)
print(f"all-0 floor: {all0_f1:.4f}")
if f1c0(yv, pv, tv.mean()) < all0_f1 + 0.05:
    print("⚠ Pipeline barely beats all-0 baseline — check signals")
"""

    # 4. Error analysis (Cell 17) - check for submission.csv or end of file
    if 'out.to_csv("submission.csv",index=False)' in source:
        if "wrong = sample.copy()" not in source:
            source += """
# ERROR ANALYSIS
wrong = sample.copy()
wrong["pred"] = (pv >= np.where(sample.no_ctx.values, tn, tc)).astype(int)
wrong["prob"] = pv
wrong = wrong[wrong["pred"] != wrong["label"]]
wrong = wrong.sort_values("prob", ascending=False)
wrong[["prompt_bn","response_bn","label","pred","prob","no_ctx"]].to_csv("/kaggle/working/errors.csv", index=False)
print(f"Wrong predictions: {len(wrong)}/{len(sample)}")
print(f"False positives (pred=1, true=0): {((wrong.pred==1)&(wrong.label==0)).sum()}")
print(f"False negatives (pred=0, true=1): {((wrong.pred==0)&(wrong.label==1)).sum()}")
"""
    
    lines = source.split('\n')
    new_source_list = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
    cell["source"] = new_source_list
    new_cells.append(cell)

nb["cells"] = new_cells
with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Applied final round of fixes!")
