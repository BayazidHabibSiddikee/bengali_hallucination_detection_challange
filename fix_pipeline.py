#!/usr/bin/env python3
"""Fix pipeline.ipynb by porting missing cells from bengali-hallu.ipynb and fixing bugs."""
import json, copy, sys

# Load both notebooks
with open("bengali-hallu.ipynb") as f:
    hallu = json.load(f)
with open("pipeline.ipynb") as f:
    pipe = json.load(f)

print(f"bengali-hallu.ipynb: {len(hallu['cells'])} cells")
print(f"pipeline.ipynb (before): {len(pipe['cells'])} cells")

def make_cell(source_lines, cell_type="code"):
    """Create a fresh code cell with no outputs."""
    if isinstance(source_lines, str):
        source_lines = [line + "\n" for line in source_lines.split("\n")]
        # Remove trailing \n from last line
        if source_lines and source_lines[-1] == "\n":
            source_lines = source_lines[:-1]
        if source_lines:
            source_lines[-1] = source_lines[-1].rstrip("\n")
    return {
        "cell_type": cell_type,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }

def clone_cell(nb, cell_idx):
    """Clone a cell from another notebook, clearing outputs."""
    cell = copy.deepcopy(nb["cells"][cell_idx])
    cell["execution_count"] = None
    cell["outputs"] = []
    cell["metadata"] = {}
    return cell

# ============================================================
# FIX 1: Replace Cell 1 (index 1) — GLOBALS → full CFG from bengali-hallu Cell 2 (index 1)
# ============================================================
print("\n[FIX 1] Replacing Cell 1 (GLOBALS) with full CFG dataclass + imports from bengali-hallu")
pipe["cells"][1] = clone_cell(hallu, 1)
print("  ✓ CFG dataclass, all imports (json, random, glob, nn, F, time, TfidfVectorizer, etc.)")

# ============================================================
# FIX 2: Fix hardcoded ==2516 in Cell 4 (index 3)
# ============================================================
print("\n[FIX 2] Fixing hardcoded ==2516 in Cell 4")
cell4 = pipe["cells"][3]
src = cell4["source"] if isinstance(cell4["source"], list) else [cell4["source"]]
new_src = []
for line in src:
    if "len(sub)==len(test)==2516" in line:
        line = line.replace("len(sub)==len(test)==2516", "len(sub)==len(test)")
        print(f"  ✓ Replaced 'len(sub)==len(test)==2516' → 'len(sub)==len(test)'")
    new_src.append(line)
cell4["source"] = new_src
cell4["outputs"] = []
cell4["execution_count"] = None

# ============================================================
# FIX 3: Insert Cell 8 (data assembly) — from bengali-hallu index 7
# Currently pipeline goes: Cell 7 (cloze, index 6) → Cell 9 (dataset/focal, index 7)
# We need to insert Cell 8 between them
# ============================================================
print("\n[FIX 3] Inserting Cell 8 (ASSEMBLE + MODE-STRATIFIED 50/50 BALANCE)")
cell8 = clone_cell(hallu, 7)
pipe["cells"].insert(7, cell8)
print("  ✓ Inserted after Cell 7 (cloze) and before Cell 9 (dataset/focal)")
# After this insert, indices shift: old index 7→8, 8→9, etc.

# ============================================================
# FIX 4: Insert Cell 15.5 (pseudo-label retrain) — from bengali-hallu index 16
# In the updated pipeline, Cell 15 (LightGBM) is now at index 15 (after insert)
# Cell 16 (submission) is at index 16. We insert 15.5 between them.
# ============================================================
print("\n[FIX 4] Inserting Cell 15.5 (PSEUDO-LABEL RETRAIN)")
cell155 = clone_cell(hallu, 16)
pipe["cells"].insert(16, cell155)
print("  ✓ Inserted after Cell 15 (LightGBM) and before Cell 16 (submission)")
# After this insert, indices shift again

# ============================================================
# FIX 5: Insert Cell 17.5 (export pseudo-labels) — from bengali-hallu index 19
# In the updated pipeline, Cell 17 (diagnostics) is now at index 19 (after 2 inserts)
# We insert after it
# ============================================================
print("\n[FIX 5] Inserting Cell 17.5 (EXPORT PSEUDO-LABELS)")
cell175 = clone_cell(hallu, 19)
pipe["cells"].insert(20, cell175)
print("  ✓ Inserted after Cell 17 (diagnostics) and before Cell 18 (visualizations)")

# ============================================================
# FIX 6: Clear all stale outputs and execution counts
# ============================================================
print("\n[FIX 6] Clearing all stale outputs and execution counts")
for i, cell in enumerate(pipe["cells"]):
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
        # Clear stale execution timestamps
        if "execution" in cell.get("metadata", {}):
            del cell["metadata"]["execution"]
        if "trusted" in cell.get("metadata", {}):
            del cell["metadata"]["trusted"]
print("  ✓ All cells reset to fresh state")

# ============================================================
# FIX 7: Update notebook metadata for Kaggle
# ============================================================
print("\n[FIX 7] Updating notebook metadata")
pipe["metadata"]["kaggle"] = {
    "accelerator": "nvidiaTeslaT4",
    "dataSources": [],
    "dockerImageVersionId": 28755,
    "isInternetEnabled": True,
    "language": "python",
    "sourceType": "notebook",
    "isGpuEnabled": True,
}
print("  ✓ Set GPU=T4, internet=True")

# ============================================================
# Verify final cell order
# ============================================================
print(f"\npipeline.ipynb (after): {len(pipe['cells'])} cells")
print("\nFinal cell order:")
for i, cell in enumerate(pipe["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell.get("source", []))
        first_line = src.split("\n")[0][:90]
        print(f"  Cell {i:2d}: {first_line}")

# Write the fixed notebook
with open("pipeline.ipynb", "w") as f:
    json.dump(pipe, f, ensure_ascii=False, indent=1)
print("\n✅ pipeline.ipynb saved successfully!")
