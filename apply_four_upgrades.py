#!/usr/bin/env python3
"""Apply four pipeline upgrades to pipeline.ipynb."""
import json
import textwrap

NOTEBOOK = "/home/sword/bengali_hallucination_detection/pipeline.ipynb"

with open(NOTEBOOK, encoding="utf-8") as f:
    nb = json.load(f)


def set_cell_source(idx, source):
    lines = source.split("\n")
    nb["cells"][idx]["source"] = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])


def find_cell(marker):
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code" and marker in "".join(cell.get("source", [])):
            return i
    raise ValueError(f"cell not found: {marker}")


def to_source_list(text):
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])


# --- CELL 1: extra installs ---
i1 = find_cell("CELL 1")
src1 = "".join(nb["cells"][i1]["source"])
src1 = src1.replace(
    'for p in ["transformers>=4.44","sentencepiece","accelerate>=0.30","bitsandbytes","datasets==2.19.0","tqdm","lightgbm"]:',
    'for p in ["transformers>=4.44","sentencepiece","accelerate>=0.30","bitsandbytes","datasets==2.19.0","tqdm","lightgbm","sentence-transformers","faiss-cpu"]:',
)
nb["cells"][i1]["source"] = to_source_list(src1)

# --- CELL 2: config knobs ---
i2 = find_cell("CELL 2")
src2 = "".join(nb["cells"][i2]["source"])
if "retr_embed_id" not in src2:
    src2 = src2.replace(
        "    n_boot:int=200\n",
        "    n_boot:int=200\n"
        "    retr_embed_id:str=\"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\"\n"
        "    chunk_size:int=500; chunk_overlap:int=150\n"
        "    use_lgbm_blend:bool=True\n"
        "    pseudo_label_n:int=500; pseudo_conf:float=0.99\n"
        "    pseudo_label_path:str=\"/kaggle/input/datasets/bayazidhs/pseudo-labels/pseudo_labels.csv\"\n",
    )
nb["cells"][i2]["source"] = to_source_list(src2)

# --- CELL 8: load pseudo labels from prior run ---
i8 = find_cell("CELL 8")
cell8 = """# ===== CELL 8 — ASSEMBLE + MODE-STRATIFIED 50/50 BALANCE =====
if len(nli_df): nli_df=nli_df.assign(mode=nli_df["src"])

def load_pseudo_labels():
    for p in (getattr(cfg, "pseudo_label_path", ""), "/kaggle/working/pseudo_labels.csv"):
        if p and os.path.exists(p):
            df = pd.read_csv(p)
            if {"premise", "response", "label"}.issubset(df.columns):
                out = df[["premise", "response", "label"]].copy()
                out["mode"] = out.get("mode", "pseudo") if "mode" in df.columns else "pseudo"
                out["src"] = out.get("src", "test_set") if "src" in df.columns else "test_set"
                print(f"Loaded {len(out)} pseudo-labels from {p}")
                return out
    return pd.DataFrame(columns=["premise", "response", "label", "mode", "src"])

parts=[d for d in (qa_df,synth_df,nli_df,load_pseudo_labels()) if d is not None and len(d)]
train_all=pd.concat([p[["premise","response","label","mode","src"]] for p in parts],ignore_index=True).dropna()
train_all=train_all[train_all["response"].str.len()>0].drop_duplicates(subset=["premise","response"])
train_all=train_all.sample(frac=1,random_state=SEED).reset_index(drop=True)

def cap(df):
    # Prioritize keeping all real QA and BHE datasets
    keep=[df[df.src.isin(["qa", "bhe_qa", "bhe_qa_full", "test_set"])]]
    room=cfg.max_train_rows-len(keep[0])
    for s in ("synth","nli","ixnli"):
        part=df[df.src==s]
        keep.append(part.sample(min(len(part),max(0,room)),random_state=SEED)); room-=len(keep[-1])
    return pd.concat(keep).sample(frac=1,random_state=SEED).reset_index(drop=True)

train_all=cap(train_all)
c1=train_all[train_all.label==1]; c0=train_all[train_all.label==0]
k=min(len(c1),len(c0))
if len(c0)>k:
    c0=(c0.groupby("mode",group_keys=False)
          .apply(lambda g:g.sample(max(1,int(round(k*len(g)/len(train_all[train_all.label==0])))),random_state=SEED)))
    c0=c0.sample(min(len(c0),k),random_state=SEED)
if len(c1)>k: c1=c1.sample(k,random_state=SEED)
train_all=pd.concat([c1,c0]).sample(frac=1,random_state=SEED).reset_index(drop=True)

n_hold=min(3000,len(train_all)//10)
synth_hold=train_all.iloc[:n_hold].reset_index(drop=True)
train_main=train_all.iloc[n_hold:].reset_index(drop=True)
print("train:",train_main.shape,"| labels:",train_main.label.value_counts().to_dict())"""
set_cell_source(i8, cell8)

# --- CELL 9: train_backbone returns model ---
i9 = find_cell("CELL 9")
src9 = "".join(nb["cells"][i9]["source"])
src9 = src9.replace(
    "    val_preds = predict_proba(model, tok, val, cfg.max_len, cfg.batch_size*2)\n"
    "    model = model.cpu()\n"
    "    del model, opt, scaler, ld, crit\n"
    "    import gc; gc.collect(); torch.cuda.empty_cache()\n"
    "    return val_preds, tok",
    "    del opt, scaler, ld, crit\n"
    "    import gc; gc.collect(); torch.cuda.empty_cache()\n"
    "    return model, tok",
)
nb["cells"][i9]["source"] = to_source_list(src9)

# --- CELL 11: FAISS + dense embeddings + overlap ---
cell11 = r'''# ===== CELL 11 — RETRIEVAL-AUGMENTED no_context (`retr`) — FAISS + Dense Embeddings =====
retr_sim_val = np.full(len(sample), np.nan)
retr_sim_test = np.full(len(test), np.nan)

def build_retr_signal():
    global retr_sim_val, retr_sim_test
    if not (cfg.use_retrieval and wiki_passages and keep_for_retr):
        return np.full(len(sample), np.nan), np.full(len(test), np.nan)

    chunks = []
    chunk_size, overlap = cfg.chunk_size, cfg.chunk_overlap
    step = chunk_size - overlap
    for p in wiki_passages:
        for i in range(0, max(1, len(p) - overlap), step):
            c = p[i:i + chunk_size]
            if len(c) > 120:
                chunks.append(c)
        if len(chunks) >= cfg.n_passages:
            break
    print(f"retrieval corpus: {len(chunks)} passages (size={chunk_size}, overlap={overlap})")

    from sentence_transformers import SentenceTransformer
    import faiss

    embed_path = resolve_model("paraphrase-multilingual-MiniLM", cfg.retr_embed_id)
    embed = SentenceTransformer(embed_path, device=DEVICE)

    embs = []
    batch = 128
    for i in range(0, len(chunks), batch):
        embs.append(
            embed.encode(
                chunks[i:i + batch],
                batch_size=batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )
    Mx = np.vstack(embs).astype(np.float32)
    d = Mx.shape[1]

    index = None
    if torch.cuda.is_available():
        try:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, d)
            print("FAISS: GPU flat inner-product index")
        except Exception as e:
            print("FAISS GPU unavailable, using CPU:", str(e)[:80])
    if index is None:
        index = faiss.IndexFlatIP(d)
        print("FAISS: CPU flat inner-product index")
    index.add(Mx)

    model, tok = keep_for_retr

    def score(df):
        out = np.full(len(df), np.nan)
        sim_out = np.full(len(df), np.nan)
        idx = np.where(df["no_ctx"].values)[0]
        if len(idx) == 0:
            return out, sim_out

        sub = df.iloc[idx]
        prompts = sub["prompt_bn"].astype(str).tolist()
        q_embs = embed.encode(
            prompts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        D, I = index.search(q_embs, cfg.retr_topk)

        prem, resp = [], []
        for ri, (r_, ti, sims) in enumerate(zip(sub.itertuples(), I, D)):
            sim_out[idx[ri]] = float(sims[0])
            for j in ti:
                prem.append(str(r_.prompt_bn) + " " + chunks[j])
                resp.append(str(r_.response_bn))

        pp = predict_proba(
            model,
            tok,
            pd.DataFrame({"premise": prem, "response": resp}),
            cfg.max_len,
            cfg.batch_size * 2,
        )
        scores_2d = pp.reshape(len(idx), cfg.retr_topk)

        MIN_SIM = 0.05
        weights = np.array([1 / (i + 1) for i in range(cfg.retr_topk)])
        for ri, (ti_row, sim_row) in enumerate(zip(I, D)):
            valid_mask = sim_row >= MIN_SIM
            if not valid_mask.any():
                valid_mask[0] = True
            w = weights * valid_mask
            scores_2d[ri] = scores_2d[ri] * (w / w.sum())

        out[idx] = scores_2d.sum(1)
        return out, sim_out

    rv, retr_sim_val = score(sample)
    rt, retr_sim_test = score(test)
    del embed, index, Mx
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rv, rt

retr_val, retr_test = build_retr_signal()
if keep_for_retr:
    del keep_for_retr
    gc.collect()
    torch.cuda.empty_cache()'''
set_cell_source(find_cell("CELL 11"), cell11)

# --- CELL 14: metadata features for LGBM ---
cell14 = r'''# ===== CELL 14 — RANK-NORMALIZE SIGNALS + META FEATURES (val∪test) =====
SIGNAL_COLS = ("enc", "lex", "retr", "llm")

def tfidf_prompt_ctx_sim(df):
    sim = np.full(len(df), np.nan)
    mask = ~df["no_ctx"].values
    if mask.sum() == 0:
        return sim
    sub = df.loc[mask]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20000, sublinear_tf=True)
    P = vec.fit_transform(sub["prompt_bn"].astype(str))
    C = vec.transform(sub["ctx_clean"].astype(str))
    sim[mask] = np.asarray(P.multiply(C).sum(axis=1)).ravel()
    return sim

def stackX(df, sv, lex, retr, llm, retr_sim=None):
    X = pd.DataFrame({c: sv[c] if c in sv else np.nan for c in SIGNAL_COLS})
    X["no_ctx"] = df["no_ctx"].values
    return X

def add_meta_features(df, X, retr_sim=None):
    X = X.copy()
    X["prompt_len"] = df["prompt_bn"].astype(str).str.len().values
    X["ctx_len"] = df["ctx_clean"].astype(str).str.len().values
    X["resp_len"] = df["response_bn"].astype(str).str.len().values
    X["tfidf_sim"] = tfidf_prompt_ctx_sim(df)
    if retr_sim is not None:
        X["retr_sim"] = retr_sim
    return X

def rank_norm(Xv, Xt):
    Xv, Xt = Xv.copy(), Xt.copy()
    for reg in (False, True):
        mv = Xv["no_ctx"].values == reg
        mt = Xt["no_ctx"].values == reg
        for c in SIGNAL_COLS:
            allv = np.concatenate([Xv.loc[mv, c].values, Xt.loc[mt, c].values]).astype(float)
            ok = ~np.isnan(allv)
            if ok.sum() < 10:
                continue
            ref = np.sort(allv[ok])

            def r(x):
                y = x.astype(float)
                m = ~np.isnan(y)
                y[m] = np.searchsorted(ref, y[m], side="right") / len(ref)
                return y

            Xv.loc[mv, c] = r(Xv.loc[mv, c].values)
            Xt.loc[mt, c] = r(Xt.loc[mt, c].values)
    return Xv, Xt

Xv = stackX(sample, sig_val, lex_val, retr_val, llm_val, retr_sim_val)
Xt = stackX(test, sig_test, lex_test, retr_test, llm_test, retr_sim_test)
Xv, Xt = rank_norm(Xv, Xt)
Xv = add_meta_features(sample, Xv, retr_sim_val)
Xt = add_meta_features(test, Xt, retr_sim_test)
yv = sample["label"].values
print("signals:", [c for c in Xv.columns if c != "no_ctx"])'''
set_cell_source(find_cell("CELL 14"), cell14)

# --- CELL 15: LightGBM meta-model ---
cell15 = r'''# ===== CELL 15 — LIGHTGBM META-MODEL STACKING (replaces Powell blender) =====
import lightgbm as lgb

def f1c0(yy, p, t):
    return f1_score(yy, (p >= t).astype(int), pos_label=0)

FEAT_COLS = [c for c in Xv.columns if c != "no_ctx"]

def tune_threshold(p, y, n_boot=cfg.n_boot, seed=SEED):
    m = len(y)
    rng = np.random.RandomState(seed)
    grid = np.quantile(p, np.linspace(0.05, 0.95, 60))
    picks = []
    for _ in range(n_boot):
        b = rng.randint(0, m, m)
        pb, yb = p[b], y[b]
        pred0 = (pb[:, None] < grid[None, :]).astype(np.float32)
        tp = ((yb == 0)[:, None] * pred0).sum(0)
        f1 = 2 * tp / np.maximum(pred0.sum(0) + (yb == 0).sum(), 1e-9)
        picks.append(grid[int(f1.argmax())])
    return float(np.median(picks))

def fit_lgbm(X, y, mask, seed=SEED):
    Xr = X.loc[mask, FEAT_COLS]
    yr = y[mask]
    dtrain = lgb.Dataset(Xr, label=yr)
    params = dict(
        objective="binary",
        metric="binary_logloss",
        verbosity=-1,
        seed=seed,
        learning_rate=0.05,
        num_leaves=15,
        min_data_in_leaf=3,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
    )
    model = lgb.train(params, dtrain, num_boost_round=150)
    p = model.predict(Xr)
    t = tune_threshold(p, yr)
    return model, t, f1c0(yr, p, t)

def lgbm_predict(X, model):
    return model.predict(X[FEAT_COLS])

mask_ctx = ~sample["no_ctx"].values
mask_no = sample["no_ctx"].values
lgb_ctx, tc, fc = fit_lgbm(Xv, yv, mask_ctx)
lgb_noctx, tn, fn = fit_lgbm(Xv, yv, mask_no)

pv = np.zeros(len(sample))
pv[mask_ctx] = lgbm_predict(Xv.loc[mask_ctx], lgb_ctx)
pv[mask_no] = lgbm_predict(Xv.loc[mask_no], lgb_noctx)
tv = np.where(sample["no_ctx"].values, tn, tc)

pt = np.zeros(len(test))
pt[~test["no_ctx"].values] = lgbm_predict(Xt.loc[~test["no_ctx"].values], lgb_ctx)
pt[test["no_ctx"].values] = lgbm_predict(Xt.loc[test["no_ctx"].values], lgb_noctx)
tt = np.where(test["no_ctx"].values, tn, tc)

THR_SHIFT = 0.0
tc = float(np.clip(tc + THR_SHIFT, 0.05, 0.95))
tn = float(np.clip(tn + THR_SHIFT, 0.05, 0.95))
tv = np.where(sample["no_ctx"].values, tn, tc)
tt = np.where(test["no_ctx"].values, tn, tc)

print("LGBM has_ctx thr", round(tc, 3), "pointF1", round(fc, 4))
print("LGBM no_ctx  thr", round(tn, 3), "pointF1", round(fn, 4))
print(
    "OVERALL valF1(c0):",
    round(f1_score(yv, (pv >= tv).astype(int), pos_label=0), 4),
    "| all-0 floor:",
    round(f1_score(yv, np.zeros(len(yv)), pos_label=0), 4),
)

imp_ctx = pd.Series(lgb_ctx.feature_importance(), index=FEAT_COLS).sort_values(ascending=False)
imp_no = pd.Series(lgb_noctx.feature_importance(), index=FEAT_COLS).sort_values(ascending=False)
print("top features has_ctx:", {k: int(v) for k, v in imp_ctx.head(5).items()})
print("top features no_ctx:", {k: int(v) for k, v in imp_no.head(5).items()})'''
set_cell_source(find_cell("CELL 15"), cell15)

# --- Insert CELL 15.5 pseudo-label retrain after cell 15 ---
cell155 = r'''# ===== CELL 15.5 — PSEUDO-LABEL RETRAIN (BanglaBERT round 2) =====
if cfg.pseudo_label_n > 0:
    test_pseudo = test.copy()
    test_pseudo["prob"] = pt
    test_pseudo["conf"] = np.maximum(test_pseudo["prob"], 1 - test_pseudo["prob"])
    confident = test_pseudo[test_pseudo["conf"] >= cfg.pseudo_conf]
    top = confident.nlargest(cfg.pseudo_label_n, "conf")

    if len(top) > 0:
        print(f"Pseudo-labeling {len(top)} test rows (conf >= {cfg.pseudo_conf})")
        pseudo_rows = pd.DataFrame(
            {
                "premise": top["premise"].values,
                "response": top["response"].values,
                "label": (top["prob"].values >= 0.5).astype(int),
                "mode": "pseudo",
                "src": "test_set",
            }
        )
        hybrid = pd.concat([train_main, pseudo_rows], ignore_index=True).drop_duplicates(
            subset=["premise", "response"]
        )

        bb_key, bb_path = cfg.backbones[0]
        print(f"Retraining {bb_key} on {len(hybrid)} rows (+{len(pseudo_rows)} pseudo)")
        m_pseudo, tk_pseudo = train_backbone(bb_key, bb_path, hybrid, sample, seed=SEED + 99)

        sig_val[bb_key] = predict_proba(m_pseudo, tk_pseudo, sample, cfg.max_len, cfg.batch_size * 2)
        sig_test[bb_key] = predict_proba(m_pseudo, tk_pseudo, test, cfg.max_len, cfg.batch_size * 2)
        enc_keys = [k for k, _ in cfg.backbones if k in sig_val]
        sig_val["enc"] = np.mean([sig_val[k] for k in enc_keys], axis=0)
        sig_test["enc"] = np.mean([sig_test[k] for k in enc_keys], axis=0)
        print(
            f"[enc round-2] val F1(c0)@0.5 = "
            f"{f1_score(sample['label'], (sig_val['enc'] >= 0.5).astype(int), pos_label=0):.4f}"
        )

        Xv = stackX(sample, sig_val, lex_val, retr_val, llm_val, retr_sim_val)
        Xt = stackX(test, sig_test, lex_test, retr_test, llm_test, retr_sim_test)
        Xv, Xt = rank_norm(Xv, Xt)
        Xv = add_meta_features(sample, Xv, retr_sim_val)
        Xt = add_meta_features(test, Xt, retr_sim_test)

        lgb_ctx, tc, fc = fit_lgbm(Xv, yv, mask_ctx, seed=SEED + 1)
        lgb_noctx, tn, fn = fit_lgbm(Xv, yv, mask_no, seed=SEED + 2)

        pv = np.zeros(len(sample))
        pv[mask_ctx] = lgbm_predict(Xv.loc[mask_ctx], lgb_ctx)
        pv[mask_no] = lgbm_predict(Xv.loc[mask_no], lgb_noctx)
        tv = np.where(sample["no_ctx"].values, tn, tc)

        pt = np.zeros(len(test))
        pt[~test["no_ctx"].values] = lgbm_predict(Xt.loc[~test["no_ctx"].values], lgb_ctx)
        pt[test["no_ctx"].values] = lgbm_predict(Xt.loc[test["no_ctx"].values], lgb_noctx)
        tt = np.where(test["no_ctx"].values, tn, tc)

        pseudo_rows.to_csv("/kaggle/working/pseudo_labels.csv", index=False)
        torch.save(m_pseudo.state_dict(), f"/kaggle/working/{bb_key}_pseudo.pt")
        del m_pseudo
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Round-2 valF1(c0): {f1_score(yv, (pv >= tv).astype(int), pos_label=0):.4f}")
    else:
        print(f"No test rows met pseudo-label confidence >= {cfg.pseudo_conf}")
else:
    print("Pseudo-label retrain disabled (cfg.pseudo_label_n=0)")'''

i15 = find_cell("CELL 15")
pseudo_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": to_source_list(cell155),
}
# insert only once
if not any("CELL 15.5" in "".join(c.get("source", [])) for c in nb["cells"]):
    nb["cells"].insert(i15 + 1, pseudo_cell)

# --- CELL 16: use LGBM pt/tt ---
i16 = find_cell("CELL 16")
cell16 = r'''# ===== CELL 16 — SUBMISSION =====
# pt/tt from Cell 15 (LightGBM meta-model); refined in Cell 15.5 if pseudo-retrain ran
out=pd.DataFrame({"id":test["id"].values,"label":(pt>=tt).astype(int)})
assert list(out.columns)==["id","label"] and len(out)==2516
assert out["label"].isin([0,1]).all() and (out["id"].values==test["id"].values).all()
out.to_csv("submission.csv",index=False)
print("submission.csv",out.shape,"| halluc rate:",round((out.label==0).mean(),3)); tleft()
# ERROR ANALYSIS
wrong = sample.copy()
wrong["pred"] = (pv >= tv).astype(int)
wrong["prob"] = pv
wrong = wrong[wrong["pred"] != wrong["label"]]
wrong = wrong.sort_values("prob", ascending=False)
wrong[["prompt_bn","response_bn","label","pred","prob","no_ctx"]].to_csv("/kaggle/working/errors.csv", index=False)
print(f"Wrong predictions: {len(wrong)}/{len(sample)}")
print(f"False positives (pred=1, true=0): {((wrong.pred==1)&(wrong.label==0)).sum()}")
print(f"False negatives (pred=0, true=1): {((wrong.pred==0)&(wrong.label==1)).sum()}")'''
set_cell_source(i16, cell16)

# --- CELL 17.5: update pseudo export text ---
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "CELL 17.5" in src and "Powell Optimizer" in src:
        src = src.replace(
            "test_pseudo['prob'] = pt  # 'pt' comes from the Powell Optimizer final blend",
            "test_pseudo['prob'] = pt  # from LightGBM meta-model (Cell 15/15.5)",
        )
        src = src.replace(
            "# Filter for extreme confidence (>95% sure it's faithful, <5% sure it's hallucination)",
            "# Filter for extreme confidence (top pseudo candidates for next-run dataset)",
        )
        cell["source"] = to_source_list(src)
        break

# --- CELL 18: feature importance viz instead of Powell weights ---
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "CELL 18" in src and "Ensemble Weights" in src:
        src = src.replace(
            "    # 2. Ensemble Weights Comparison\n"
            "    if 'wc' in globals() and 'wn' in globals():\n"
            "        weights_df = pd.DataFrame({\n"
            "            'Feature': list(wc.keys()),\n"
            "            'Has Context': list(wc.values()),\n"
            "            'No Context': list(wn.values())\n"
            "        })\n"
            "        fig_weights = px.bar(weights_df, x='Feature', y=['Has Context', 'No Context'], barmode='group',\n"
            "                             title=\"Optimized Ensemble Weights (Has Context vs No Context)\",\n"
            "                             color_discrete_sequence=[\"#636EFA\", \"#FFA15A\"])\n"
            "        fig_weights.show()\n",
            "    # 2. LightGBM Feature Importance\n"
            "    if 'imp_ctx' in globals() and 'imp_no' in globals():\n"
            "        weights_df = pd.DataFrame({\n"
            "            'Feature': imp_ctx.index.tolist(),\n"
            "            'Has Context': imp_ctx.values,\n"
            "            'No Context': imp_no.reindex(imp_ctx.index).fillna(0).values,\n"
            "        })\n"
            "        fig_weights = px.bar(weights_df, x='Feature', y=['Has Context', 'No Context'], barmode='group',\n"
            "                             title=\"LightGBM Meta-Model Feature Importance\",\n"
            "                             color_discrete_sequence=[\"#636EFA\", \"#FFA15A\"])\n"
            "        fig_weights.show()\n",
        )
        cell["source"] = to_source_list(src)
        break

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Applied all four upgrades to pipeline.ipynb")
