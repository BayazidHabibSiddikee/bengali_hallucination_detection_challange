import json

new_cell_11 = """# ===== CELL 11 — RETRIEVAL-AUGMENTED no_context (`retr`) — FAISS + Dense =====
retr_sim_val  = np.full(len(sample), np.nan)
retr_sim_test = np.full(len(test),   np.nan)

def build_retr_signal():
    global retr_sim_val, retr_sim_test

    # ── Safe keep_for_retr access ─────────────────────────────────────────────
    _kfr = globals().get("keep_for_retr", None)

    # Retrieval still works without BanglaBERT reranker — uses sim scores directly
    if not cfg.use_retrieval or not wiki_passages:
        print("⚠ Retrieval disabled or no wiki passages — skipping")
        return np.full(len(sample), np.nan), np.full(len(test), np.nan)

    # ── Build passage chunks ──────────────────────────────────────────────────
    chunks     = []
    chunk_size = cfg.chunk_size
    overlap    = cfg.chunk_overlap
    step       = chunk_size - overlap

    for p in wiki_passages:
        for i in range(0, max(1, len(p) - overlap), step):
            c = p[i:i + chunk_size]
            if len(c) > 120:
                chunks.append(c)
        if len(chunks) >= cfg.n_passages:
            break
    print(f"retrieval corpus: {len(chunks)} passages "
          f"(size={chunk_size}, overlap={overlap})")

    if not chunks:
        print("⚠ No chunks built — skipping retrieval")
        return np.full(len(sample), np.nan), np.full(len(test), np.nan)

    # ── Build FAISS index ─────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    import faiss

    embed_path = resolve_model("paraphrase-multilingual-MiniLM", cfg.retr_embed_id)
    embed      = SentenceTransformer(embed_path, device=DEVICE)

    embs = []
    for i in range(0, len(chunks), 128):
        embs.append(embed.encode(
            chunks[i:i + 128],
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ))
    Mx = np.vstack(embs).astype(np.float32)
    d  = Mx.shape[1]

    index = None
    if torch.cuda.is_available():
        try:
            res   = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, d)
            print("FAISS: GPU flat inner-product index")
        except Exception as e:
            print("FAISS GPU unavailable, using CPU:", str(e)[:80])
    if index is None:
        index = faiss.IndexFlatIP(d)
        print("FAISS: CPU flat inner-product index")
    index.add(Mx)
    print(f"FAISS index built: {index.ntotal} vectors, dim={d}")

    # ── Score one dataframe ───────────────────────────────────────────────────
    def score(df):
        out     = np.full(len(df), np.nan)
        sim_out = np.full(len(df), np.nan)

        idx = np.where(df["no_ctx"].values)[0]
        if len(idx) == 0:
            return out, sim_out

        sub     = df.iloc[idx]
        prompts = sub["prompt_bn"].astype(str).tolist()
        q_embs  = embed.encode(
            prompts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        D, I = index.search(q_embs, cfg.retr_topk)

        MIN_SIM = 0.05
        weights = np.array([1 / (i + 1) for i in range(cfg.retr_topk)],
                           dtype=np.float32)

        # ── Branch A: BanglaBERT reranking available ──────────────────────
        if _kfr is not None:
            model, tok = _kfr
            prem, resp = [], []
            for ri, (r_, ti, sims) in enumerate(
                    zip(sub.itertuples(), I, D)):
                sim_out[idx[ri]] = float(sims[0])
                for j in ti:
                    prem.append(str(r_.prompt_bn) + " " + chunks[j])
                    resp.append(str(r_.response_bn))

            pp = predict_proba(
                model, tok,
                pd.DataFrame({"premise": prem, "response": resp}),
                cfg.max_len, cfg.batch_size * 2,
            )
            scores_2d = pp.reshape(len(idx), cfg.retr_topk).copy()

            for ri, sim_row in enumerate(D):
                valid = sim_row >= MIN_SIM
                if not valid.any(): valid[0] = True
                w = weights * valid
                scores_2d[ri] = scores_2d[ri] * (w / w.sum())

            out[idx] = scores_2d.sum(1)

        # ── Branch B: no reranker — use raw FAISS similarity ─────────────
        else:
            print("  ℹ No BanglaBERT reranker — using FAISS similarity scores")
            for ri, (sim_row, score_row) in enumerate(zip(D, I)):
                sim_out[idx[ri]] = float(sim_row[0])
                valid = sim_row >= MIN_SIM
                if not valid.any(): valid[0] = True
                w = weights * valid
                out[idx[ri]] = float((sim_row * (w / w.sum())).sum())

        return out, sim_out

    # ── Run on val and test ───────────────────────────────────────────────────
    rv, retr_sim_val  = score(sample)
    rt, retr_sim_test = score(test)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    del embed, index, Mx
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    has_retr = (~np.isnan(rv)).sum()
    print(f"✅ Retrieval done: {has_retr}/{len(sample)} val rows scored "
          f"({'with' if _kfr else 'without'} BanglaBERT reranking)")
    return rv, rt


retr_val, retr_test = build_retr_signal()

# Safe delete keep_for_retr
if globals().get("keep_for_retr") is not None:
    del globals()["keep_for_retr"]
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ keep_for_retr released")
"""

with open("pipeline.ipynb") as f:
    nb = json.load(f)

replaced = False
for c in nb["cells"]:
    if c.get("cell_type") == "code":
        src = "".join(c.get("source", []))
        if "CELL 11 — RETRIEVAL-AUGMENTED" in src:
            c["source"] = new_cell_11.splitlines(keepends=True)
            replaced = True
            break

if replaced:
    with open("pipeline.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print("SUCCESS")
else:
    print("FAILED")
