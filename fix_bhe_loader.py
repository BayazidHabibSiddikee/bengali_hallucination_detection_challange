import json

NEW_LOAD_BHE = '''def load_bhe_datasets():
    """
    Load BanglaHalluEval QA datasets.

    Two file formats exist:
    1. banglahallueval_qa_1000.csv    → cols: id, context, question, correct_answer
       (Pure QA: generate hallucinations via augmentation)
    2. banglahallueval_qa_dataset*.csv → cols: id, question, deepseek_answer, gemma_answer,
       qwen_answer, correct_answer, deepseek_score, gemma_score, qwen_score
       (LLM outputs scored 0=hallucinated, 1=faithful)
    """
    rows = []
    base = "/kaggle/input/datasets/mahdihasanqurishi/banglahallueval-qa"

    # --- File 1: banglahallueval_qa_1000.csv (context+question+correct_answer) ---
    p1 = f"{base}/banglahallueval_qa_1000.csv"
    if os.path.exists(p1):
        try:
            df = pd.read_csv(p1)
            all_answers = df["correct_answer"].dropna().astype(str).tolist()
            for _, r in df.iterrows():
                ctx = str(r.get("context", "")).strip()
                if ctx.lower() in {"nan","null","none",""}: ctx = ""
                q = str(r.get("question", "")).strip()
                ans = str(r.get("correct_answer", "")).strip()
                if not q or not ans or ans.lower() == "nan": continue
                prem = (q + " " + ctx).strip() if ctx else q
                # Faithful row
                rows.append((prem, ans, 1, "bhe_qa", "bhe_qa"))
                # Hallucinated: pick a different answer from the pool
                import random as _random
                others = [a for a in all_answers if a != ans]
                if others:
                    rows.append((prem, _random.choice(others), 0, "bhe_hallucinated", "bhe_qa"))
        except Exception as e:
            print(f"BHE qa_1000 load error: {e}")

    # --- File 2: banglahallueval_qa_dataset*.csv (LLM scored outputs) ---
    for fname in ("banglahallueval_qa_dataset.csv", "banglahallueval_qa_dataset_1000.csv"):
        p = f"{base}/{fname}"
        if not os.path.exists(p): continue
        try:
            df = pd.read_csv(p)
            # Column mapping: each LLM gets its score column
            llm_pairs = [
                ("deepseek_answer", "deepseek_score"),
                ("gemma_answer",    "gemma_score"),
                ("qwen_answer",     "qwen_score"),
            ]
            for _, r in df.iterrows():
                q = str(r.get("question", "")).strip()
                if not q: continue
                correct = str(r.get("correct_answer", "")).strip()
                ctx = ""  # These files have no context column
                prem = q

                # Add the correct answer as faithful
                if correct and correct.lower() != "nan":
                    rows.append((prem, correct, 1, "bhe_faithful", "bhe_qa"))

                # Add each scored LLM answer
                for ans_col, score_col in llm_pairs:
                    if ans_col not in r or score_col not in r: continue
                    ans = str(r[ans_col]).strip()
                    score = r[score_col]
                    if not ans or ans.lower() == "nan": continue
                    try:
                        lbl = int(float(score))
                        if lbl not in (0, 1): lbl = 0  # default hallu if ambiguous
                    except:
                        continue
                    if ans != correct:  # avoid duplicating the correct answer
                        rows.append((prem, ans, lbl, "bhe_llm_scored", "bhe_qa"))
        except Exception as e:
            print(f"BHE {fname} load error: {e}")

    if not rows:
        return pd.DataFrame(columns=["premise", "response", "label", "mode", "src"])

    out = pd.DataFrame(rows, columns=["premise", "response", "label", "mode", "src"])
    out = out.drop_duplicates(subset=["premise", "response"])
    print(f"  BHE loaded {len(out)} rows | modes: {out['mode'].value_counts().to_dict()}")
    return out
'''

def fix_notebook(filename):
    with open(filename) as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src_str = ''.join(cell.get('source', []))
        if 'def load_bhe_datasets' in src_str:
            new_src = []
            skip = False
            replaced = False
            for line in cell.get('source', []):
                if 'def load_bhe_datasets()' in line and not replaced:
                    skip = True
                    # Inject new function
                    for new_line in NEW_LOAD_BHE.split('\n'):
                        new_src.append(new_line + '\n')
                    replaced = True
                    continue
                if skip:
                    # End of old function: next top-level statement
                    if line.strip() and not line.startswith(' ') and not line.startswith('\t') and not line.startswith('#'):
                        skip = False
                        new_src.append(line)
                    continue
                new_src.append(line)
            nb['cells'][i]['source'] = new_src
            print(f"Fixed load_bhe_datasets in {filename} cell {i}")

    with open(filename, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

fix_notebook('pipeline.ipynb')
fix_notebook('bengali-hallu.ipynb')
