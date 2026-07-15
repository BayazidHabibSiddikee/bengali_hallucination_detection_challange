import json, re

def fix(filename):
    with open(filename) as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src_str = ''.join(cell.get('source', []))
        if 'backbones:tuple' in src_str:
            # Find and replace just the backbones line(s) cleanly
            new_src = []
            skip_next = False
            for line in cell.get('source', []):
                if 'backbones:tuple=' in line:
                    # Replace entire backbones definition with single-model version
                    new_src.append('    backbones:tuple=((\"banglabert_large\",\"csebuetnlp/banglabert_large\"),)  # mDeBERTa disabled\n')
                    skip_next = True
                    continue
                if skip_next:
                    # Skip any continuation lines from the old tuple
                    if line.strip().startswith('#') or 'mdeberta' in line.lower() or line.strip().startswith(')'):
                        continue
                    else:
                        skip_next = False
                        new_src.append(line)
                    continue
                new_src.append(line)
            nb['cells'][i]['source'] = new_src
            print(f"Fixed backbones in {filename} cell {i}")

    json.dumps(nb)  # validate
    with open(filename, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

fix('pipeline.ipynb')
fix('bengali-hallu.ipynb')

# Final verify
import json
with open('pipeline.ipynb') as f:
    nb = json.load(f)
src = ''.join(nb['cells'][1].get('source', []))
idx = src.find('backbones:tuple')
print('\nFinal backbones:')
print(src[idx:idx+100])
