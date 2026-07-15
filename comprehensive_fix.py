import json, copy

with open('pipeline.ipynb') as f:
    nb = json.load(f)

print("Current cell count:", len(nb['cells']))
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell.get('source', []))
        print(f"Cell {i}: {src[:80].replace(chr(10),' ')}")

