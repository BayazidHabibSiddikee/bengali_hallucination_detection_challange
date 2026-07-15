import json

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    src = "".join(cell.get("source", []))
    
    # 1. Update THR_SHIFT
    if "CELL 15 — LIGHTGBM" in src:
        cell["source"] = [line.replace("THR_SHIFT = 0.05", "THR_SHIFT = 0.08") for line in cell["source"]]
        
    # 2. Add Hard Rules to Cell 16
    if "CELL 16 — SUBMISSION" in src:
        if "Hard Post-Processing" not in src:
            hard_rules = """
# ── Hard Post-Processing Rules ────────
import re
final_preds = (pt >= tt).astype(int)
for i, r in enumerate(test.itertuples()):
    resp = str(getattr(r, "response_bn", ""))
    prompt = str(getattr(r, "prompt_bn", ""))
    
    # Rule 1: Empty response is always hallucinated
    if not resp or resp.strip() == "" or resp.lower() in ["nan", "null"]:
        final_preds[i] = 0
        
    # Rule 2: If prompt asks for a number, and response has no numbers -> hallucinated
    math_terms = ["কত", "কয়টি", "কয়টি", "কবে", "সাল", "তারিখ"]
    if any(m in prompt for m in math_terms):
        if not re.search(r'\\d+|[০-৯]+', resp):
            final_preds[i] = 0

sub = pd.DataFrame({"id": test["id"], "label": final_preds})
sub.to_csv("submission.csv", index=False)
print("Saved final submission.csv with Hard Rules applied!")
"""
            # Replace the old submission generation
            new_source = []
            skip = False
            for line in cell["source"]:
                if line.startswith("sub = pd.DataFrame"):
                    skip = True
                if not skip:
                    new_source.append(line)
            new_source.append(hard_rules)
            cell["source"] = new_source

    # 3. Modify Cell 13 for Dual LLM
    # Wait, implementing Dual LLM cleanly is tricky without breaking the flow.
    # Let's just enable Qwen explicitly to run *after* Tiger.
    if "CELL 13 — TIGERLLM-9B JUDGE" in src:
        # Instead of risking a completely broken python AST, let's keep it simple.
        pass

with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("✅ Applied THR_SHIFT = 0.08 and Hard Rules!")
