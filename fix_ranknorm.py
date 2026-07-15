import json

def fix_notebook(filename):
    with open(filename) as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = cell.get('source', [])
            new_src = []
            skip = False
            for line in src:
                # Remove rank_norm definition and calls
                if 'def rank_norm(' in line:
                    skip = True
                if skip and line.strip() == 'return Xv, Xt':
                    skip = False
                    continue
                if skip:
                    continue
                
                if 'Xv, Xt = rank_norm(Xv, Xt)' in line:
                    continue
                
                new_src.append(line)
            
            nb['cells'][i]['source'] = new_src

    with open(filename, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Removed rank_norm from {filename}")

fix_notebook('pipeline.ipynb')
fix_notebook('bengali-hallu.ipynb')
