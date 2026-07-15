import json

with open('pipeline.ipynb') as f:
    nb = json.load(f)

print("=" * 60)
print("FINAL COMPETITION READINESS AUDIT")
print("=" * 60)

errors = []
warnings = []
ok = []

# Get all code
cells = {}
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        cells[i] = ''.join(cell.get('source', []))

all_src = '\n'.join(cells.values())

# ===== CRITICAL CHECKS =====

# 1. JSON validity
try:
    json.dumps(nb)
    ok.append("JSON structure is valid")
except Exception as e:
    errors.append(f"JSON INVALID: {e}")

# 2. No hardcoded row counts
if '== 2516' in all_src or '==2516' in all_src:
    errors.append("Hardcoded == 2516 row count found! Will break on Phase 2!")
else:
    ok.append("No hardcoded row counts")

# 3. No hardcoded paths outside /kaggle/
import re
bad_paths = re.findall(r'(?<!#)["\'](?!/kaggle|/tmp|/usr|/opt|/root|http)[\'"]', all_src)
local_paths = [p for p in re.findall(r'["\']/(home|tmp|var|etc)[^"\']*["\']', all_src)]
if local_paths:
    warnings.append(f"Local paths found (won't work on Kaggle): {local_paths[:3]}")
else:
    ok.append("No local filesystem paths")

# 4. HF_TOKEN set in env
if 'os.environ["HF_TOKEN"]' in all_src or "os.environ['HF_TOKEN']" in all_src:
    ok.append("HF_TOKEN exported to os.environ (fixes download crashes)")
else:
    errors.append("HF_TOKEN not exported to os.environ!")

# 5. No rank_norm
if 'def rank_norm' in all_src or 'rank_norm(Xv' in all_src:
    errors.append("rank_norm still present! This was the main score killer!")
else:
    ok.append("rank_norm removed (was distorting probabilities)")

# 6. z_score_norm present everywhere needed
zscore_count = all_src.count('z_score_norm(Xv, Xt)')
if zscore_count >= 2:
    ok.append(f"z_score_norm called {zscore_count} times (Round 1 + Round 2 both covered)")
elif zscore_count == 1:
    warnings.append("z_score_norm only called once - Round 2 may have scale mismatch")
else:
    errors.append("z_score_norm MISSING!")

# 7. StratifiedKFold present
if 'StratifiedKFold' in all_src:
    ok.append("StratifiedKFold cross-validation present")
else:
    errors.append("StratifiedKFold missing - threshold tuning may overfit!")

# 8. Category feature for C1 tiebreaker
if 'category_enc' in all_src:
    ok.append("category_enc feature present (helps C1 cultural subset)")
else:
    warnings.append("category_enc missing - C1 tiebreaker performance not optimized")

# 9. Checkpoint loading logic
if 'skipping training' in all_src and 'find_ckpt' in all_src:
    ok.append("Checkpoint loading (skip training when .pt available)")
else:
    errors.append("Checkpoint loading logic missing!")

# 10. Nuclear clear memory management
if 'nuclear_clear' in all_src:
    ok.append("nuclear_clear() memory management present")
else:
    errors.append("nuclear_clear() missing - LLM may OOM!")

# 11. Submission format check
if 'sub.columns' in all_src and 'id' in all_src and '"label"' in all_src:
    ok.append("Submission format check present")
else:
    warnings.append("No submission format validation")

# 12. No duplicate torch.save
ts_count = all_src.count('torch.save(m.state_dict()')
if ts_count == 1:
    ok.append(f"torch.save called exactly once")
elif ts_count > 1:
    warnings.append(f"torch.save called {ts_count} times (may waste time)")

# 13. THR_SHIFT applied
if 'THR_SHIFT' in all_src:
    shift_val = re.search(r'THR_SHIFT\s*=\s*([\d\.\-]+)', all_src)
    val = shift_val.group(1) if shift_val else '?'
    ok.append(f"THR_SHIFT applied (value={val}) for leaderboard probe")
else:
    warnings.append("THR_SHIFT missing")

# 14. test_signals.csv exported
if 'test_signals.csv' in all_src:
    ok.append("test_signals.csv exported (enables offline LightGBM loop)")
else:
    warnings.append("test_signals.csv not exported")

# 15. Pseudo-label confidence threshold
pl_match = re.search(r'pseudo_conf\s*[:=]\s*([\d\.]+)', all_src)
if pl_match:
    pconf = float(pl_match.group(1))
    if pconf >= 0.99:
        ok.append(f"pseudo_conf={pconf} (strict — prevents noisy pseudo-labels)")
    else:
        warnings.append(f"pseudo_conf={pconf} may be too low — risky pseudo-labels")

# 16. assert len(sub)==len(test)  (flexible, no hardcoded ==2516)
if 'assert list(sub.columns)' in all_src and '==2516' not in all_src:
    ok.append("Submission assertion is flexible (no hardcoded row count)")

# 17. Cell count
if len(nb['cells']) == 21:
    ok.append(f"Correct 21-cell architecture")
else:
    warnings.append(f"Cell count is {len(nb['cells'])} (expected 21)")

# 18. max_train_rows set reasonably
mr_match = re.search(r'max_train_rows\s*[:=]\s*(\d+)', all_src)
if mr_match:
    mr = int(mr_match.group(1))
    ok.append(f"max_train_rows={mr}")

# Print results
print(f"\n✅ PASSING ({len(ok)} checks):")
for o in ok:
    print(f"   ✅ {o}")

if warnings:
    print(f"\n⚠️  WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"   ⚠️  {w}")

if errors:
    print(f"\n❌ ERRORS ({len(errors)}) — MUST FIX BEFORE SUBMIT:")
    for e in errors:
        print(f"   ❌ {e}")
else:
    print(f"\n{'='*60}")
    print("🏆 NO CRITICAL ERRORS — READY TO SUBMIT!")
    print(f"{'='*60}")

