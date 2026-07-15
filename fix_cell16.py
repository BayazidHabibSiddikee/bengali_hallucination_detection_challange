import json

def fix_cell16(filename):
    with open(filename, 'r') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = cell.get('source', [])
            if any('Xv = add_meta_features(sample, Xv, retr_sim_val)' in line for line in src) and any('bb_key, bb_path = cfg.backbones[0]' in line for line in src):
                new_src = []
                for line in src:
                    new_src.append(line)
                    if 'Xt = add_meta_features(test, Xt, retr_sim_test)' in line:
                        new_src.append('        Xv, Xt = z_score_norm(Xv, Xt)\n')
                
                nb['cells'][i]['source'] = new_src

    with open(filename, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Fixed Cell 16 data leakage in {filename}")

fix_cell16('pipeline.ipynb')
fix_cell16('bengali-hallu.ipynb')
