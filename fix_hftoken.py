import json

def fix_token(filename):
    with open(filename, 'r') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = cell.get('source', [])
            new_src = []
            for line in src:
                new_src.append(line)
                if 'HF_TOKEN=get_hf_token()' in line:
                    new_src.append('if HF_TOKEN:\n')
                    new_src.append('    os.environ["HF_TOKEN"] = HF_TOKEN\n')
            
            nb['cells'][i]['source'] = new_src

    with open(filename, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Fixed HF_TOKEN in {filename}")

fix_token('pipeline.ipynb')
fix_token('bengali-hallu.ipynb')
