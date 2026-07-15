import json
import re

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    
    # CFG edits
    if "class CFG:" in source:
        # Change batch_size to 8 if not already
        source = re.sub(r'batch_size\s*:\s*int\s*=\s*\d+', 'batch_size:int=8', source)
        source = re.sub(r'batch_size\s*=\s*\d+', 'batch_size = 8', source)
        
        if "max_mem_llm" not in source:
            # Add to class
            source = source.replace('llm_id:str="4bit"', 'llm_id:str="4bit"\n    max_mem_llm:dict = None\n    llm_input_len:int = 512')
            
        if "cfg.max_mem_llm =" not in source:
            source = source.replace('cfg=CFG()', 'cfg=CFG()\ncfg.max_mem_llm = {0: "3GiB", 1: "13GiB", "cpu": "40GiB"}')

    # Data loading edits (Cell 4 / Cell 2 in user terms)
    if "dataset samples.json" in source and "test set.csv" in source:
        old_loading = """sample=pd.DataFrame(json.load(open(os.path.join(cfg.comp_dir,"/kaggle/input/datasets/bayazidhs/bengali-hallucination-data/dataset samples.json"),encoding="utf-8")))
test=pd.read_csv(os.path.join(cfg.comp_dir,"/kaggle/input/datasets/bayazidhs/bengali-hallucination-data/test set.csv"))
sub=pd.read_csv(os.path.join(cfg.comp_dir,"/kaggle/input/datasets/bayazidhs/bengali-hallucination-data/sample submission.csv"))"""
        
        new_loading = """SAMPLE_PATH = "/kaggle/input/datasets/bayazidhs/bengali-hallucination-data/dataset samples.json"
TEST_PATH   = "/kaggle/input/datasets/bayazidhs/bengali-hallucination-data/test set.csv"
SUB_PATH    = "/kaggle/input/datasets/bayazidhs/bengali-hallucination-data/sample submission.csv"

sample = pd.DataFrame(json.load(open(SAMPLE_PATH, encoding="utf-8")))
test   = pd.read_csv(TEST_PATH)
sub    = pd.read_csv(SUB_PATH)

if "id" not in test.columns:
    test.insert(0, "id", range(len(test)))
    print("⚠ Added synthetic id column to test")"""
        source = source.replace(old_loading, new_loading)
        
    # NLI Edits
    if "load_indicxnli" in source:
        # Remove neutral NLI
        source = source.replace('if p and n: rows.append((p, n, 0, "nli"))', '# Neutral dropped')
        # Cap to 15k
        source = source.replace('sample(min(40000,len(df)))', 'sample(min(15000, len(df)), random_state=SEED)')

    lines = source.split('\n')
    new_source_list = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
    cell["source"] = new_source_list
    
with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    
print("Second round of fixes applied.")
