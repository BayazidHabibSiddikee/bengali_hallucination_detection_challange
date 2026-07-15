import json

def fix_find_ckpt(filename):
    with open(filename) as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src_str = ''.join(cell.get('source', []))
        if 'def find_ckpt' in src_str:
            # Rebuild this function cleanly
            new_src = []
            in_fn = False
            replaced = False
            for line in cell.get('source', []):
                if 'def find_ckpt(key):' in line:
                    in_fn = True
                    new_src.extend([
                        'def find_ckpt(key):\n',
                        '    # Scans multiple locations for a pre-trained .pt checkpoint\n',
                        '    patterns = [\n',
                        '        f"/kaggle/input/datasets/bayazidhs/trained-banglabert/{key}.pt",\n',
                        '        f"/kaggle/input/datasets/bayazidhs/trained-banglabert/{key}/{key}.pt",\n',
                        '        f"/kaggle/input/**/{key}.pt",\n',
                        '    ]\n',
                        '    for pat in patterns:\n',
                        '        hits = glob.glob(pat, recursive=True)\n',
                        '        if hits: return hits[0]\n',
                        '    return None\n',
                        '\n',
                    ])
                    replaced = True
                    continue

                if in_fn:
                    # skip old function body until we hit next function/code block
                    if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                        in_fn = False
                        new_src.append(line)
                    # skip all old function lines
                    continue

                new_src.append(line)

            if replaced:
                nb['cells'][i]['source'] = new_src
                print(f"Fixed find_ckpt in {filename} cell {i}")

    with open(filename, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

fix_find_ckpt('pipeline.ipynb')
fix_find_ckpt('bengali-hallu.ipynb')
