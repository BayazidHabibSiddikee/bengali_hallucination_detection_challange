import json

def update_colab():
    with open('adding_multifold_training.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            if 'def find_file(pattern):' in source:
                # Update find_file to take a filename and search both environments
                old_func = '''def find_file(pattern):
    hits = glob.glob(pattern, recursive=True)
    return hits[0] if hits else None

SAMPLE_PATH = find_file("/content/**/dataset samples.json")
NLI_TSV     = find_file("/content/**/NLI Dataset - Combined.tsv")
WIKI_DIR    = find_file("/content/**/bengali-wikipedia-articles/*/train")
BHE_FULL    = find_file("/content/**/banglahallueval_qa_dataset.csv")
PSEUDO_PATH = find_file("/content/**/pseudo_labels.csv")'''
                
                new_func = '''def find_file(filename):
    for base in ["/kaggle/input", "/content"]:
        hits = glob.glob(f"{base}/**/{filename}", recursive=True)
        if hits: return hits[0]
    return None

SAMPLE_PATH = find_file("dataset samples.json")
NLI_TSV     = find_file("NLI Dataset - Combined.tsv")
WIKI_DIR    = find_file("bengali-wikipedia-articles/*/train")
if not WIKI_DIR: WIKI_DIR = find_file("bengali-wikipedia-articles/train/train")
BHE_FULL    = find_file("banglahallueval_qa_dataset.csv")
PSEUDO_PATH = find_file("pseudo_labels.csv")'''
                
                source = source.replace(old_func, new_func)
                
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source'] and cell['source'][-1].endswith('\n\n'):
                cell['source'][-1] = cell['source'][-1][:-1]
                
    with open('adding_multifold_training.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    update_colab()
