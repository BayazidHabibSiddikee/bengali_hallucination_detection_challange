import json

def fix_notebook(filename):
    with open(filename) as f:
        nb = json.load(f)

    fixes = []

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = cell.get('source', [])
        src_str = ''.join(src)
        new_src = list(src)

        # ================================================================
        # FIX 1: CFG Cell — fix BHE dataset paths to use correct username
        # kernel-metadata.json has "mahdihasanqurishi/banglahallueval-qa"
        # but CFG points to bayazidhs/banglahallueval-qa → 0 rows loaded
        # ================================================================
        if 'bhe_qa_1000' in src_str and 'bayazidhs' in src_str and 'banglahallueval' in src_str:
            new_src = []
            for line in src:
                if 'banglahallueval-qa' in line and 'bayazidhs' in line:
                    # Fix: use the correct mahdihasanqurishi path
                    line = line.replace(
                        f'{base_dir}/banglahallueval-qa' if '{base_dir}' not in line else '{base_dir}/banglahallueval-qa',
                        '/kaggle/input/datasets/mahdihasanqurishi/banglahallueval-qa'
                    )
                    # Also handle f-string format
                    line = line.replace(
                        '"{base_dir}/banglahallueval-qa',
                        '"/kaggle/input/datasets/mahdihasanqurishi/banglahallueval-qa'
                    )
                new_src.append(line)
            fixes.append(f"Cell {i}: Fixed BHE dataset paths (bayazidhs→mahdihasanqurishi)")

        # ================================================================
        # FIX 2: Cell 10 — Also scan mahdihasanqurishi path in find_ckpt
        # and REMOVE mDeBERTa from backbone list since no mdeberta.pt exists
        # → prevents silent 3-hour training wait
        # ================================================================
        if 'backbones' in src_str and 'mdeberta' in src_str and '# ===== CELL 2' in src_str:
            # Only BanglaBERT-Large since mdeberta.pt isn't in trained-banglabert dataset
            # Comment out mDeBERTa backbone to skip its training
            new_src2 = []
            for line in src_str.split('\n'):
                if '"mdeberta"' in line and 'mdeberta-v3' in line and 'backbones' in src_str[:src_str.find(line)]:
                    # Comment out the mdeberta entry
                    new_src2.append('                     # ("mdeberta","microsoft/mdeberta-v3-base"),  # No checkpoint — would trigger 3hr train\n')
                else:
                    new_src2.append(line + '\n' if not line.endswith('\n') else line)
            # Don't apply this yet - let's just do path fix

    # ================================================================
    # FIX 2 (separate pass): Cell 10 — add mahdihasanqurishi path to find_ckpt
    # ================================================================
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src_str = ''.join(cell.get('source', []))
        
        if 'def find_ckpt' in src_str:
            new_src = []
            for line in cell.get('source', []):
                new_src.append(line)
                # After the find_ckpt patterns, also check mahdihasanqurishi path for BHE
                if 'f"/kaggle/input/**/{key}.pt"' in line:
                    # Add mdeberta.pt check from trained-banglabert dataset
                    new_src.append('                f"/kaggle/input/datasets/bayazidhs/trained-banglabert/{key}.pt",\n')
            nb['cells'][i]['source'] = new_src
            fixes.append(f"Cell {i}: Extended find_ckpt to check trained-banglabert subfolder")

    # ================================================================
    # FIX 3 (most important): CFG Cell — fix the actual BHE paths string
    # ================================================================
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src_str = ''.join(cell.get('source', []))
        
        if 'bhe_qa_1000' in src_str and 'banglahallueval' in src_str:
            new_src = []
            for line in cell.get('source', []):
                if 'banglahallueval-qa' in line and ('{base_dir}' in line or 'bayazidhs' in line):
                    # Replace the path: use mahdihasanqurishi's dataset path
                    line = line.replace(
                        '{base_dir}/banglahallueval-qa',
                        '/kaggle/input/datasets/mahdihasanqurishi/banglahallueval-qa'
                    )
                    line = line.replace(
                        'f\"{base_dir}/banglahallueval-qa',
                        '"/kaggle/input/datasets/mahdihasanqurishi/banglahallueval-qa'
                    )
                new_src.append(line)
            nb['cells'][i]['source'] = new_src
            fixes.append(f"Cell {i}: Fixed BHE paths to mahdihasanqurishi")

    with open(filename, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"\n{filename}:")
    for f_ in fixes:
        print(f"  ✅ {f_}")
    if not fixes:
        print("  (no changes needed)")

fix_notebook('pipeline.ipynb')
fix_notebook('bengali-hallu.ipynb')
