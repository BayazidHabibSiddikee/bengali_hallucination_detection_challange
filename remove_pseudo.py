import json

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# The cells to delete by their index
# Cell 17 (index 16) is PSEUDO-LABEL RETRAIN
# Cell 20 (index 19) is EXPORT PSEUDO-LABELS
indices_to_delete = []
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if "CELL 15.5 — PSEUDO-LABEL" in src or "CELL 17.5 — EXPORT PSEUDO-LABELS" in src:
        indices_to_delete.append(i)

# Delete in reverse order so indices don't shift
for i in sorted(indices_to_delete, reverse=True):
    del nb["cells"][i]

with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"✅ Removed {len(indices_to_delete)} pseudo-labeling cells to comply with competition rules!")
