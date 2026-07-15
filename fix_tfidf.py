import json

def fix_tfidf():
    with open('pipeline.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "def tfidf_prompt_ctx_sim(df):" in source:
                # We need to change the signature and logic
                old_func = """def tfidf_prompt_ctx_sim(df):
    sim = np.full(len(df), np.nan)
    mask = ~df["no_ctx"].values
    if mask.sum() == 0:
        return sim
    sub = df.loc[mask]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20000, sublinear_tf=True)
    P = vec.fit_transform(sub["prompt_bn"].astype(str))
    C = vec.transform(sub["ctx_clean"].astype(str))
    sim[mask] = np.asarray(P.multiply(C).sum(axis=1)).ravel()
    return sim"""
                
                new_func = """def tfidf_prompt_ctx_sim(df, vec=None):
    sim = np.full(len(df), np.nan)
    mask = ~df["no_ctx"].values
    if mask.sum() == 0:
        return sim, vec
    sub = df.loc[mask]
    if vec is None:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20000, sublinear_tf=True)
        # Fit on both prompt and context for a comprehensive vocabulary
        vec.fit(sub["prompt_bn"].astype(str).tolist() + sub["ctx_clean"].astype(str).tolist())
    P = vec.transform(sub["prompt_bn"].astype(str))
    C = vec.transform(sub["ctx_clean"].astype(str))
    sim[mask] = np.asarray(P.multiply(C).sum(axis=1)).ravel()
    return sim, vec"""
                
                source = source.replace(old_func, new_func)
                
                old_meta = """def add_meta_features(df, X, retr_sim=None):"""
                new_meta = """def add_meta_features(df, X, retr_sim=None, tfidf_vec=None):"""
                source = source.replace(old_meta, new_meta)
                
                old_call = """    X["tfidf_sim"] = tfidf_prompt_ctx_sim(df)"""
                new_call = """    sim_res, tfidf_vec_out = tfidf_prompt_ctx_sim(df, tfidf_vec)
    X["tfidf_sim"] = sim_res"""
                source = source.replace(old_call, new_call)
                
                old_return = """    return X"""
                new_return = """    return X, tfidf_vec_out"""
                source = source.replace(old_return, new_return)
                
                old_apply = """Xv = add_meta_features(sample, Xv, retr_sim_val)
Xt = add_meta_features(test, Xt, retr_sim_test)"""
                new_apply = """Xv, fitted_tfidf = add_meta_features(sample, Xv, retr_sim_val, None)
Xt, _ = add_meta_features(test, Xt, retr_sim_test, fitted_tfidf)"""
                source = source.replace(old_apply, new_apply)
                
                cell['source'] = [line + '\n' for line in source.split('\n')]
                if cell['source'] and cell['source'][-1].endswith('\n\n'):
                    cell['source'][-1] = cell['source'][-1][:-1]
                
    with open('pipeline.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    fix_tfidf()
