import json

with open('/home/sword/bengali_hallucination_detection/pipeline.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if "PDF_DIR = '/home/sword/Documents/tools/downloads'" in src:
            src = src.replace("PDF_DIR = '/home/sword/Documents/tools/downloads'", "PDF_DIR = '/kaggle/input/bengali-historical-books'")
            src = src.replace("FAISS_DB_PATH = '/home/sword/bengali_hallucination_detection/faiss_db'", "FAISS_DB_PATH = '/kaggle/working/faiss_db'")
            
            # Since Kaggle kernels might not have rank_bm25 and faiss-cpu by default, let's inject a pip install at the top
            cell['source'] = ["!pip install -q faiss-cpu rank_bm25 lightgbm langchain-community langchain-huggingface sentence-transformers pypdf\n\n"] + [src]
        elif cell['source']:
            cell['source'] = [src]

with open('/home/sword/bengali_hallucination_detection/pipeline.ipynb', 'w') as f:
    json.dump(nb, f)
