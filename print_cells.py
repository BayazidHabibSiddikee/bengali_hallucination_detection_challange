import json
with open("pipeline.ipynb") as f:
    nb = json.load(f)

for i, c in enumerate(nb["cells"]):
    if i >= 13 and c.get("cell_type") == "code":
        sep = "="*40
        print(f"\n{sep}\nCELL {i+1}\n{sep}")
        src = "".join(c.get("source", []))
        for j, line in enumerate(src.splitlines()):
            print(f"{j+1:3d}: {line}")
