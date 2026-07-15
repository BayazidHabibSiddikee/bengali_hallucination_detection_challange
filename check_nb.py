import json

with open("pipeline.ipynb") as f:
    nb = json.load(f)

cells = nb.get("cells", [])
print(f"Total cells: {len(cells)}")
print()
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    print(f"Cell {i+1} | {c.get('cell_type')} | {len(src)} chars | {src[:80].replace(chr(10), ' ')}")
