#!/usr/bin/env python3
"""Closes the two worthwhile gaps vs the original architecture diagram:
1. LEAKAGE AUDIT (Stage 0): duplicate-id check + exact (prompt,response) matches
   between the labeled sample and the test set; exact matches get their KNOWN
   label copied into the submission (free accuracy, zero risk).
2. REGIME FEATURES (Router -> meta): is_math / is_translation / is_mcq flags and
   a Bengali-numeral-aware number_support score for the LightGBM stacker, so the
   router's regimes reach the meta-model instead of stopping at has_ctx/no_ctx.
"""
import json

NB = "pipeline.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]

def find_cell(marker):
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            return i
    raise SystemExit(f"marker not found: {marker}")

def repl(i, old, new, what):
    s = "".join(cells[i]["source"])
    assert old in s, f"anchor missing for: {what}"
    cells[i]["source"] = [s.replace(old, new)]
    print("applied:", what)

def append(i, extra, what):
    s = "".join(cells[i]["source"])
    cells[i]["source"] = [s.rstrip() + "\n" + extra]
    print("applied:", what)

# ---- 1a. leakage audit in CELL 4 ---------------------------------------------
append(find_cell("CELL 4 — COMPETITION DATA"), r'''
# --- LEAKAGE AUDIT (Stage 0 of the architecture) ---
# exact (prompt, response) matches between the labeled sample and the test set
# carry a KNOWN label — the submission cell copies it over the model prediction.
def _leak_key(p, r):
    return re.sub(r"\s+", " ", str(p).strip().lower()) + " || " + re.sub(r"\s+", " ", str(r).strip().lower())
_known = {_leak_key(p, r): int(l)
          for p, r, l in zip(sample["prompt_bn"], sample["response_bn"], sample["label"])}
test["leak_label"] = [_known.get(_leak_key(p, r), np.nan)
                      for p, r in zip(test["prompt_bn"], test["response_bn"])]
print("leakage audit | duplicate test ids:", int(test["id"].duplicated().sum()),
      "| exact sample->test matches:", int(test["leak_label"].notna().sum()))''',
"CELL 4 leakage audit")

# ---- 1b. leak-label override in the submission cell ---------------------------
i = find_cell("CELL 16 — SUBMISSION")
repl(i,
'''out=pd.DataFrame({"id":test["id"].values,"label":(pt>=tt).astype(int)})''',
'''out=pd.DataFrame({"id":test["id"].values,"label":(pt>=tt).astype(int)})
# exact duplicates of labeled sample rows get their known label (leakage audit, Cell 4)
if "leak_label" in test.columns:
    _lm = test["leak_label"].notna().values
    if _lm.any():
        out.loc[_lm, "label"] = test.loc[_lm, "leak_label"].astype(int).values
        print(f"leak override: {int(_lm.sum())} exact-match rows set to known labels")''',
"submission leak-label override")

# ---- 2. regime features for the LightGBM stacker ------------------------------
i = find_cell("CELL 14 — RANK-NORMALIZE")
s = "".join(cells[i]["source"])
anchor = '''    else:
        X["category_enc"] = 3.0  # default to general_knowledge
    return X'''
assert anchor in s, "add_meta_features tail not found"
repl(i, anchor,
'''    else:
        X["category_enc"] = 3.0  # default to general_knowledge
    # --- regime flags (Task/Regime Router -> meta-model) ---
    pr = df["prompt_bn"].astype(str); rs = df["response_bn"].astype(str)
    cx = df["ctx_clean"].astype(str)
    X["is_math"] = [int(bool(numset(p)) and bool(numset(r))) for p, r in zip(pr, rs)]
    X["is_translation"] = pr.str.contains(
        "অনুবাদ|translate|ইংরেজিতে|সারাংশ|সংক্ষেপে|summar", regex=True, case=False).astype(int).values
    X["is_mcq"] = (pr.str.contains(r"ক\\)", regex=True)
                   & pr.str.contains(r"খ\\)", regex=True)).astype(int).values
    def _numsup(p, r, c):
        # fraction of response numbers that also appear in prompt+context
        # (Bengali numerals normalized); -1 = response has no numbers
        nr = numset(r)
        return -1.0 if not nr else len(nr & (numset(p) | numset(c))) / len(nr)
    X["number_support"] = [_numsup(p, r, c) for p, r, c in zip(pr, rs, cx)]
    return X''',
"regime features in add_meta_features")

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

import ast
bad = 0
for j, c in enumerate(cells):
    if c["cell_type"] == "code":
        try:
            ast.parse("".join(c["source"]))
        except SyntaxError as e:
            bad += 1
            print(f"SYNTAX ERROR cell {j}: {e}")
print(f"done: {len(cells)} cells, {bad} syntax errors")
