#!/usr/bin/env python3
"""Executes pipeline.ipynb cell-by-cell in one namespace (no jupyter needed).
Local smoke test: uses ./dev_data + small stand-in models on CPU."""
import json
import sys
import traceback

import matplotlib
matplotlib.use("Agg")

with open("pipeline.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

ns = {"__name__": "__main__"}
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
    print(f"\n{'=' * 60}\n### CELL {i}\n{'=' * 60}", flush=True)
    try:
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

print("\nALL CELLS PASSED")
