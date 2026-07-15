import json

def fix(filename):
    with open(filename) as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src_str = ''.join(cell.get('source', []))
        if 'backbones' in src_str and 'mdeberta-v3-base' in src_str and 'CFG' in src_str:
            new_src = []
            for line in cell.get('source', []):
                # Comment out mDeBERTa backbone — only use BanglaBERT (has checkpoint)
                if '"mdeberta"' in line and 'mdeberta-v3-base' in line:
                    line = line.replace(
                        '("mdeberta","microsoft/mdeberta-v3-base"),',
                        '# ("mdeberta","microsoft/mdeberta-v3-base"),  # Disabled: no checkpoint mounted'
                    )
                    line = line.replace(
                        '("mdeberta","microsoft/mdeberta-v3-base")',
                        '# ("mdeberta","microsoft/mdeberta-v3-base")  # Disabled: no checkpoint mounted'
                    )
                new_src.append(line)
            nb['cells'][i]['source'] = new_src
            print(f"Disabled mDeBERTa in {filename} cell {i}")

    # Also verify JSON
    json.dumps(nb)

    with open(filename, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

fix('pipeline.ipynb')
fix('bengali-hallu.ipynb')
