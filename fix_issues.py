import json

def fix_both(filename):
    with open(filename) as f:
        nb = json.load(f)

    changed = []

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = cell.get('source', [])
        src_str = ''.join(src)

        # ======================================================
        # FIX 1: Cell 10 - Remove duplicate torch.save
        # ======================================================
        if 'torch.save(m.state_dict()' in src_str and src_str.count('torch.save(m.state_dict()') > 1:
            new_src = []
            torch_save_seen = False
            for line in src:
                if 'torch.save(m.state_dict()' in line:
                    if not torch_save_seen:
                        new_src.append(line)
                        torch_save_seen = True
                    # skip duplicate
                    else:
                        changed.append(f"[{filename}] Cell {i}: Removed duplicate torch.save")
                else:
                    new_src.append(line)
            nb['cells'][i]['source'] = new_src

        # ======================================================
        # FIX 2: Cell 13 - Save category to sample/test DataFrames
        # and also extract category as a numeric feature for LightGBM
        # The C1 "cultural-distance subset" = Bengali-specific questions
        # (vocabulary, history, general knowledge without context)
        # ======================================================
        if '# ===== CELL 13' in src_str and 'llm_val = run_llm_judge(sample)' in src_str:
            CAT_EXTRACTION = """
# ── Extract category labels for LightGBM meta-feature ──────────────────────
# C1 tie-breaker = cultural/linguistic Bengali rows (vocab, history, gk)
# We encode these as numeric so LightGBM can route them differently
CATEGORY_MAP = {"comprehension": 0, "math": 1, "vocabulary": 2,
                "general_knowledge": 3, "history": 4, "code_mixed": 5}

def _categorize_df(df):
    cats = []
    for r in df.itertuples():
        ctx = getattr(r, "ctx_clean", "")
        cats.append(get_category(r.prompt_bn, ctx, r.response_bn))
    return cats

sample["category"] = _categorize_df(sample)
test["category"]   = _categorize_df(test)
print("Val category dist:", sample["category"].value_counts().to_dict())
"""
            new_src = list(src)
            # Insert after last line of existing code, before llm_val line
            final = []
            for line in new_src:
                if 'llm_val = run_llm_judge(sample)' in line and 'sample["category"]' not in src_str:
                    final.extend([l + '\n' for l in CAT_EXTRACTION.strip().split('\n')])
                    final.append('\n')
                final.append(line)
            nb['cells'][i]['source'] = final
            changed.append(f"[{filename}] Cell {i}: Added category extraction to sample/test")

    # ======================================================
    # FIX 3: Cell 14 - add_meta_features - add category as numeric feature
    # ======================================================
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src_str = ''.join(cell.get('source', []))
        if 'def add_meta_features' in src_str and 'category_enc' not in src_str:
            new_src = []
            for line in cell.get('source', []):
                new_src.append(line)
                if 'X["n_signals_missing"]' in line:
                    # Add category encoding right after
                    new_src.extend([
                        '\n',
                        '    # Category encoding for LightGBM (C1 cultural-distance routing)\n',
                        '    CATEGORY_MAP = {"comprehension": 0, "math": 1, "vocabulary": 2,\n',
                        '                    "general_knowledge": 3, "history": 4, "code_mixed": 5}\n',
                        '    if "category" in df.columns:\n',
                        '        X["category_enc"] = df["category"].map(CATEGORY_MAP).fillna(3).values\n',
                        '    else:\n',
                        '        X["category_enc"] = 3.0  # default to general_knowledge\n',
                    ])
                    changed.append(f"[{filename}] Cell {i}: Added category_enc to add_meta_features")
            nb['cells'][i]['source'] = new_src

    with open(filename, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"\n{filename}: {len(changed)} fixes applied")
    for c in changed:
        print(f"  ✅ {c}")

fix_both('pipeline.ipynb')
fix_both('bengali-hallu.ipynb')
