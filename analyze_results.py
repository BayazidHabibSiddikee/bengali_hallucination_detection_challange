import json
import pandas as pd

try:
    with open("bengali-hallu (1).ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for i, cell in enumerate(nb.get("cells", [])):
        source = "".join(cell.get("source", []))
        if "LGBM has_ctx thr" in source or "CELL 15" in source or "valF1" in source or "print(\"LGBM" in source:
            print(f"--- Output from Cell {i} (LightGBM/Validation) ---")
            for out in cell.get("outputs", []):
                if "text" in out:
                    print("".join(out["text"]))
                
    df = pd.read_csv("submission (1).csv")
    print(f"\n--- Submission Stats ---")
    print(f"Total Rows: {len(df)}")
    print("Label Distribution:")
    print(df["label"].value_counts())
    print(f"Percentage of 1 (Faithful): {df['label'].mean()*100:.1f}%")
except Exception as e:
    print(f"Error: {e}")
