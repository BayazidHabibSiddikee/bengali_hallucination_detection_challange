import json
import re

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        
        # 1. Update max_train_rows and retr_topk
        if "max_train_rows:int=" in source:
            source = re.sub(r'max_train_rows:int=\d+', 'max_train_rows:int=25000', source)
            source = re.sub(r'retr_topk:int=\d+', 'retr_topk:int=5', source)
            
            # Add historical books path to CFG
            if "bhe_qa_ds_1000:str" in source and "books_dir:str" not in source:
                source = source.replace('bhe_qa_ds_1000:str =', 'books_dir:str = f"{base_dir}/bengali-historical-books"\n    bhe_qa_ds_1000:str =')
                
            cell["source"] = [source]
            
        # 2. Update data priority to use bhe_qa_full
        if "keep=[df[df.src.isin" in source or "def cap(df):" in source:
            source = source.replace('["qa", "bhe_qa"]', '["qa", "bhe_qa", "bhe_qa_full"]')
            source = source.replace('["qa","bhe_qa"]', '["qa","bhe_qa","bhe_qa_full"]')
            cell["source"] = [source]

        # 3. Add books to retrieval corpus (Cell 7 usually handles wiki_passages)
        if "files=glob.glob(os.path.join(cfg.wiki_dir" in source:
            source = source.replace('files=glob.glob(os.path.join(cfg.wiki_dir,"**/*.txt"),recursive=True)', 'files=glob.glob(os.path.join(cfg.wiki_dir,"**/*.txt"),recursive=True)\n    books=glob.glob(os.path.join(cfg.books_dir,"**/*.txt"),recursive=True) if hasattr(cfg, "books_dir") else []\n    files.extend(books)')
            cell["source"] = [source]
            
with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

