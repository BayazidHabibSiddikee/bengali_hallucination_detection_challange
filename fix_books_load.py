import json

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        
        if "def load_wiki(limit):" in source:
            # We want to replace the `for s in ... files+=` line to also include books
            target_line = '    for s in ("train/train","valid/valid",""): files+=glob.glob(os.path.join(cfg.wiki_dir,s,"*.txt"))'
            replacement = target_line + '\n    if hasattr(cfg, "books_dir") and type(cfg.books_dir) == str and len(cfg.books_dir) > 0: files+=glob.glob(os.path.join(cfg.books_dir, "**/*.txt"), recursive=True)'
            
            source = source.replace(target_line, replacement)
            cell["source"] = [source]
            
with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

