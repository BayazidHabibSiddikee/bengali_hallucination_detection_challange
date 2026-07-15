import json
import glob
import re

notebooks = glob.glob("top_10_notebooks/**/*.ipynb", recursive=True)

print("| Author | ML Frameworks | HuggingFace Models | External Datasets |")
print("|--------|--------------|-------------------|-------------------|")

for nb_path in notebooks:
    author = nb_path.split("/")[1]
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        
        code_cells = [cell["source"] for cell in nb.get("cells", []) if cell.get("cell_type") == "code"]
        code = "\n".join("".join(c) for c in code_cells)
        
        models = set()
        for match in re.findall(r'["\']([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)["\']', code):
            if "sentence" in match.lower() or "bert" in match.lower() or "qwen" in match.lower() or "llm" in match.lower():
                models.add(match)
                
        frameworks = set()
        if "XGB" in code or "xgb" in code: frameworks.add("XGBoost")
        if "LGBM" in code or "lgb" in code: frameworks.add("LightGBM")
        if "CatBoost" in code: frameworks.add("CatBoost")
        if "SentenceTransformer" in code: frameworks.add("SentenceTransformers")
        if "AutoModel" in code: frameworks.add("HuggingFace")
        
        datasets = set()
        if "squad" in code.lower(): datasets.add("SQuAD")
        if "tydiqa" in code.lower(): datasets.add("TyDiQA")
        if "indicqa" in code.lower(): datasets.add("IndicQA")
        
        m_str = ", ".join(models) if models else "None"
        f_str = ", ".join(frameworks) if frameworks else "None"
        d_str = ", ".join(datasets) if datasets else "None"
        
        print(f"| {author} | {f_str} | {m_str} | {d_str} |")
        
    except Exception:
        pass
