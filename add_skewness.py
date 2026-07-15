import json

def add_skew_features(filename):
    with open(filename, 'r') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = cell.get('source', [])
            if any('def add_meta_features' in line for line in src):
                new_src = []
                skip = False
                for line in src:
                    if 'def add_meta_features' in line:
                        skip = True
                        new_src.extend([
                            'from scipy.stats import skew, kurtosis\n\n',
                            'def add_meta_features(df, X, retr_sim=None):\n',
                            '    X = X.copy()\n',
                            '    X["prompt_len"] = df["prompt_bn"].astype(str).str.len().values\n',
                            '    X["ctx_len"] = df["ctx_clean"].astype(str).str.len().values\n',
                            '    X["resp_len"] = df["response_bn"].astype(str).str.len().values\n',
                            '    X["tfidf_sim"] = tfidf_prompt_ctx_sim(df)\n',
                            '    if retr_sim is not None:\n',
                            '        X["retr_sim"] = retr_sim\n\n',
                            '    CORE = ["enc", "lex", "retr", "llm"]\n',
                            '    sig_matrix = np.column_stack([\n',
                            '        X[c].fillna(0.5).values if c in X.columns else np.full(len(df), 0.5)\n',
                            '        for c in CORE\n',
                            '    ])\n',
                            '    X["signal_skew"] = skew(sig_matrix, axis=1, bias=True)\n',
                            '    X["signal_kurt"] = kurtosis(sig_matrix, axis=1, bias=True)\n',
                            '    X["signal_std"] = sig_matrix.std(axis=1)\n',
                            '    X["signal_range"] = sig_matrix.max(axis=1) - sig_matrix.min(axis=1)\n',
                            '    X["signal_max_mean_gap"] = sig_matrix.max(axis=1) - sig_matrix.mean(axis=1)\n',
                            '    X["n_signals_hallu"] = (sig_matrix < 0.5).sum(axis=1).astype(float)\n\n',
                            '    sig_raw = np.column_stack([\n',
                            '        X[c].values if c in X.columns else np.full(len(df), np.nan)\n',
                            '        for c in CORE\n',
                            '    ])\n',
                            '    X["n_signals_missing"] = np.isnan(sig_raw).sum(axis=1).astype(float)\n',
                            '    return X\n'
                        ])
                        continue
                    
                    if skip and 'Xv = stackX' in line:
                        skip = False
                    
                    if skip:
                        continue
                        
                    new_src.append(line)
                
                nb['cells'][i]['source'] = new_src

    with open(filename, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Added skew/kurtosis meta-features to {filename}")

add_skew_features('pipeline.ipynb')
add_skew_features('bengali-hallu.ipynb')
