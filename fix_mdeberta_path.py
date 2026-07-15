import json

def fix(filename):
    with open(filename) as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src_str = ''.join(cell.get('source', []))
        if 'def find_ckpt' in src_str:
            new_src = []
            for line in cell.get('source', []):
                # Replace the patterns tuple with one that includes the bengali-trained-mdeberta dataset
                if 'f"/kaggle/input/datasets/bayazidhs/trained-banglabert/{key}/{key}.pt"' in line:
                    new_src.append(line)
                    # Insert the mdeberta-specific path right after
                    new_src.append('        f"/kaggle/input/datasets/bayazidhs/bengali-trained-mdeberta/{key}.pt",\n')
                else:
                    new_src.append(line)
            nb['cells'][i]['source'] = new_src
            print(f"Updated find_ckpt in {filename} cell {i}")

    with open(filename, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

fix('pipeline.ipynb')
fix('bengali-hallu.ipynb')
