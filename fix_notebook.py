import json

# Load the top notebook
with open("top_10_notebooks/mahdihasanqurishi/gemini.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Fix the paths in CELL 2
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "comp_dir:str=" in source:
            # Change comp_dir
            source = source.replace('comp_dir:str="/kaggle/input/competitions/bengali-hallucination"', 'comp_dir:str="/kaggle/input/bengali-hallucination-data"')
            # Change the nli_tsv path to just datasets instead of specific user since Kaggle mounts datasets simply by slug sometimes, 
            # actually kaggle mounts by `<creator>/<dataset>`. The paths in the original notebook are:
            # /kaggle/input/datasets/ajmainmahtab/bangla-natural-language-inference-dataset/...
            # Actually Kaggle mounts datasets directly under /kaggle/input/<dataset-slug>/
            # Let's fix these paths to work universally on Kaggle.
            source = source.replace('/kaggle/input/datasets/ajmainmahtab/bangla-natural-language-inference-dataset', '/kaggle/input/bangla-natural-language-inference-dataset')
            source = source.replace('/kaggle/input/datasets/disisbig/bengali-wikipedia-articles', '/kaggle/input/bengali-wikipedia-articles')
            source = source.replace('/kaggle/input/datasets/mahdihasanqurishi/banglahallueval-qa', '/kaggle/input/banglahallueval-qa')
            
            cell["source"] = [source]
            break

with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

