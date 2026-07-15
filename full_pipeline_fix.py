import json

with open('pipeline.ipynb') as f:
    nb = json.load(f)

def get_src(cell_idx):
    return ''.join(nb['cells'][cell_idx].get('source', []))

def set_src(cell_idx, lines):
    nb['cells'][cell_idx]['source'] = lines if isinstance(lines, list) else [lines]

issues = []

# ========================================================
# CHECK 1: Cell 13 - TigerLLM - Is 'category' feature added to meta-features?
# The C1 "cultural-distance" subset is likely: questions about Bengali culture,
# history, vocabulary where no foreign context is needed.
# We need to add a 'category_enc' feature to the LightGBM
# ========================================================
src13 = get_src(13)
if 'get_category' not in src13:
    issues.append("Cell 13: get_category function missing!")
else:
    print("✅ Cell 13: get_category exists")

# Check if category is exported as a column  
if 'sample["category"]' not in src13 and "sample['category']" not in src13:
    issues.append("Cell 13: category not saved to sample/test DataFrames")
else:
    print("✅ Cell 13: category saved to DataFrames")

# ========================================================
# CHECK 2: Cell 14 - add_meta_features - Does it include category?
# ========================================================
src14 = get_src(14)
if 'signal_skew' not in src14:
    issues.append("Cell 14: signal_skew missing from add_meta_features")
else:
    print("✅ Cell 14: signal_skew present")

if 'z_score_norm' not in src14:
    issues.append("Cell 14: z_score_norm missing!")
else:
    print("✅ Cell 14: z_score_norm present")

# ========================================================
# CHECK 3: Cell 15 - LightGBM - duplicate line?
# ========================================================
src15 = get_src(15)
if 'lgb_noctx, tn, fn = fit_lgbm(X\nlgb_noctx, tn, fn = fit_lgbm(Xv, yv, mask_no)' in src15:
    issues.append("Cell 15: DUPLICATE broken line in lgb_noctx!")
else:
    print("✅ Cell 15: no duplicate lgb_noctx line")

# ========================================================
# CHECK 4: Cell 16 - Pseudo-label retrain - has z_score_norm?
# ========================================================
src16 = get_src(16)
if 'z_score_norm' not in src16:
    issues.append("Cell 16: Missing z_score_norm in pseudo-label retrain!")
else:
    print("✅ Cell 16: z_score_norm present")

# ========================================================
# CHECK 5: Cell 17 - Submission - uses correct threshold variable?
# ========================================================
src17 = get_src(17)
if 'tt' not in src17 and 'tv' not in src17:
    issues.append("Cell 17: submission not using threshold array (tt/tv)")
else:
    print("✅ Cell 17: submission uses threshold")

# ========================================================
# CHECK 6: Cell 2 - HF_TOKEN exported to env?
# ========================================================
src2 = get_src(1)
if 'os.environ["HF_TOKEN"]' not in src2 and "os.environ['HF_TOKEN']" not in src2:
    issues.append("Cell 2: HF_TOKEN not set in os.environ!")
else:
    print("✅ Cell 2: HF_TOKEN exported to os.environ")

# ========================================================
# CHECK 7: Cell 10 - Double torch.save?
# ========================================================
src10 = get_src(9)
if src10.count('torch.save(m.state_dict()') > 1:
    issues.append("Cell 10: DUPLICATE torch.save!")
else:
    print("✅ Cell 10: no duplicate torch.save")

# ========================================================  
# CHECK 8: Cell 18 - Diagnostics - exports test_signals.csv?
# ========================================================
src18 = get_src(18)
if 'test_signals.csv' not in src18:
    issues.append("Cell 18: Missing test_signals.csv export")
else:
    print("✅ Cell 18: test_signals.csv exported")

print()
print("=" * 50)
print(f"ISSUES FOUND: {len(issues)}")
for issue in issues:
    print(f"  ❌ {issue}")
