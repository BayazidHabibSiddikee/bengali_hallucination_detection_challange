import json
import glob
import os
import re

notebooks = glob.glob("top_10_notebooks/**/*.ipynb", recursive=True)

for nb_path in notebooks:
    print(f"\n{'='*50}\nAnalyzing: {nb_path}\n{'='*50}")
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        
        code_cells = [cell["source"] for cell in nb.get("cells", []) if cell.get("cell_type") == "code"]
        code = "\n".join("".join(c) for c in code_cells)
        
        # Look for model names
        hf_models = set(re.findall(r'["\']([^"\']*/[^"\']*)["\']', code))
        hf_models = {m for m in hf_models if '/' in m and len(m.split('/')) == 2 and not m.startswith('/')}
        
        # Look for libraries
        libs = set(re.findall(r'^\s*(?:import|from)\s+([a-zA-Z0-9_]+)', code, flags=re.MULTILINE))
        
        print(f"Libraries used: {', '.join(sorted(libs))}")
        print(f"Potential HF Models: {', '.join(sorted(hf_models))}")
        
        if "XGB" in code or "xgb" in code: print("- Uses XGBoost")
        if "LGBM" in code or "lgb" in code: print("- Uses LightGBM")
        if "CatBoost" in code: print("- Uses CatBoost")
        if "SentenceTransformer" in code: print("- Uses SentenceTransformers")
        if "AutoModel" in code: print("- Uses HuggingFace AutoModel")
        if "prompt" in code.lower(): print("- Uses prompts (likely LLM based)")
        
    except Exception as e:
        print(f"Error reading {nb_path}: {e}")
