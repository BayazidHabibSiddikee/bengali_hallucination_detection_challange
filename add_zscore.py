import json

def apply_zscore(filename):
    with open(filename) as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = cell.get('source', [])
            if any('def add_meta_features' in line for line in src):
                new_src = []
                for line in src:
                    new_src.append(line)
                    if 'def add_meta_features' in line:
                        zscore_code = """
def z_score_norm(Xv, Xt):
    Xv, Xt = Xv.copy(), Xt.copy()
    for c in Xv.columns:
        if c == "no_ctx": continue
        # Calculate mean/std ONLY on validation set to prevent data leakage from the test set
        mean = Xv[c].mean()
        std = Xv[c].std()
        if std == 0 or np.isnan(std):
            std = 1.0
        Xv[c] = (Xv[c] - mean) / std
        Xt[c] = (Xt[c] - mean) / std
    return Xv, Xt
"""
                        # Insert zscore code right before add_meta_features
                        for zline in zscore_code.split('\n'):
                            if zline or True: # keep empty lines
                                new_src.insert(-1, zline + '\n')
                
                # Now add the call to z_score_norm at the end
                final_src = []
                for line in new_src:
                    final_src.append(line)
                    if 'Xt = add_meta_features(test, Xt, retr_sim_test)' in line:
                        final_src.append('Xv, Xt = z_score_norm(Xv, Xt)\n')
                
                nb['cells'][i]['source'] = final_src

    with open(filename, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Added Z-Score normalization to {filename}")

apply_zscore('pipeline.ipynb')
apply_zscore('bengali-hallu.ipynb')
