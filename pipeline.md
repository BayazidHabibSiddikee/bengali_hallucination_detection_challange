```python
# ===== CELL 1 — INSTALLS (datasets PINNED: tydiqa/squad_bn are script datasets, broken on 3.x) =====
import subprocess, sys, shutil
for p in ["transformers>=4.44","sentencepiece","accelerate>=0.30","bitsandbytes","datasets==2.19.0","tqdm","lightgbm","sentence-transformers"]:
    subprocess.run([sys.executable,"-m","pip","install","-q",p],check=False)
# "faiss-gpu" has no pip wheel for Kaggle's python — ensure faiss-cpu is importable.
# (CPU flat inner-product search over 250k x 384 vectors takes milliseconds.)
try:
    import faiss
    print("ok | faiss already available")
except ImportError:
    subprocess.run([sys.executable,"-m","pip","install","-q","faiss-cpu"],check=False)
    print("ok | installed faiss-cpu")




```

```python
# ===== CELL 2 — CONFIG · SEEDS · SECRETS =====
import os, re, gc, glob, json, random, unicodedata, warnings, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
warnings.filterwarnings("ignore"); T0=time.time()
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

def get_hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None
HF_TOKEN=get_hf_token()
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

@dataclass
class CFG:
    base_dir = "/kaggle/input/datasets/bayazidhs"
    comp_dir:str = f"{base_dir}/bengali-hallucination-data"
    nli_tsv:str = f"/kaggle/input/datasets/ajmainmahtab/bangla-natural-language-inference-dataset/NLI Dataset - Combined.tsv"
    wiki_dir:str = f"/kaggle/input/datasets/disisbig/bengali-wikipedia-articles"

    bhe_qa_1000:str = f"/kaggle/input/datasets/mahdihasanqurishi/banglahallueval-qa/banglahallueval_qa_1000.csv"
    bhe_qa_full:str = f"/kaggle/input/datasets/mahdihasanqurishi/banglahallueval-qa/banglahallueval_qa_dataset.csv"
    books_dir:str = f"{base_dir}/bengali-historical-books"
    bhe_qa_ds_1000:str = f"/kaggle/input/datasets/mahdihasanqurishi/banglahallueval-qa/banglahallueval_qa_dataset_1000.csv"

    hf_squad:str="csebuetnlp/squad_bn"
    hf_tydi:tuple=("tydiqa","google-research-datasets/tydiqa")
    hf_ixnli:str="Divyanshu/indicxnli"
    hf_qa_70k:str="rasheduzzaman/Bangla_question_answer_pair_70K_dataset"

    local_squad:str=""; local_tydi:str=""; local_ixnli:str=""; local_qa_70k:str=""
    # Encoder ensemble: each backbone trains (or loads an attached .pt checkpoint)
    # and their probabilities are averaged into one "enc" signal before blending.
    # Swap "microsoft/mdeberta-v3-base" for "xlm-roberta-large" if time allows
    # (xlm-r-large is ~1.7x slower to train, slightly stronger on Bengali).
    backbones:tuple=(("banglabert_large","csebuetnlp/banglabert_large"),)  # mDeBERTa disabled
    llm_id:str="md-nishat-008/TigerLLM-9B-it"
    max_len:int=256; batch_size:int=8; epochs:int=4; lr:float=8e-6; warmup:float=0.15
    focal_gamma:float=2.0; focal_alpha:float=0.75
    n_wiki_files:int=8000; max_train_rows:int=25000
    use_retrieval:bool=True; n_passages:int=250000; retr_topk:int=5
    use_llm_judge:bool=True; judge_dual_prompt:bool=False
    n_boot:int=200
    retr_embed_id:str="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size:int=500; chunk_overlap:int=150
    use_lgbm_blend:bool=True
    pseudo_label_n:int=500; pseudo_conf:float=0.99
    pseudo_label_path:str="/kaggle/input/datasets/bayazidhs/pseudo-labels/pseudo_labels.csv"
    
cfg=CFG()
cfg.max_mem_llm = {0: "3GiB", 1: "13GiB", "cpu": "40GiB"}
def tleft(): print(f"[t+{(time.time()-T0)/60:.1f}m]")
print("device:",DEVICE,"| gpus:",torch.cuda.device_count(),"| HF token:",("set" if HF_TOKEN else "None (ok)"))




```

```python
# ===== CELL 3 — BENGALI UTILS =====
BN_DIGITS="০১২৩৪৫৬৭৮৯"; BN2ASCII={ord(b):str(i) for i,b in enumerate(BN_DIGITS)}
NULLS={"[null]","null","none","nan","n/a",""}; _P=set("।,.?!;:\"'()[]{}<>/\\|-–—’‘“”…%°")
BN_CHAR=re.compile(r"[\u0980-\u09FF]"); DIGIT_RE=re.compile(r"[০-৯0-9]+")
def norm(s): return unicodedata.normalize("NFC",str(s))
def denum(s): return norm(s).translate(BN2ASCII)
def is_no_ctx(c):
    if c is None or (isinstance(c,float) and pd.isna(c)): return True
    return str(c).strip().lower() in NULLS
def toks(s): return [t for t in "".join(" " if c in _P else c for c in denum(s)).split() if t]
def numset(s): return set(re.findall(r"\d+(?:\.\d+)?",denum(s)))
def content(s,m=2): return {t for t in toks(s) if len(t)>=m}
def contain(resp,src):
    r,s=content(resp),content(src); return len(r&s)/len(r) if r else 1.0
def sent_split(t):
    parts=re.split(r"(?<=[।!?])\s+|\n+",norm(t))
    return [p.strip() for p in parts if len(p.strip())>8]
def mostly_bengali(s):
    s=str(s); b=len(BN_CHAR.findall(s)); return b>=max(1,int(0.5*len(re.findall(r"\S",s))))
def bump_digits(a):
    def b(ch):
        if ch in BN_DIGITS: return BN_DIGITS[(BN_DIGITS.index(ch)+random.randint(1,8))%10]
        if ch.isdigit(): return str((int(ch)+random.randint(1,8))%10)
        return ch
    n="".join(b(c) for c in a); return None if n==a else n




```

```python
# ===== CELL 4 — COMPETITION DATA =====
SAMPLE_PATH = "/kaggle/input/datasets/bayazidhs/bengali-hallucination-data/dataset samples.json"
TEST_PATH   = "/kaggle/input/datasets/bayazidhs/bengali-hallucination-data/test set.csv"
SUB_PATH    = "/kaggle/input/datasets/bayazidhs/bengali-hallucination-data/sample submission.csv"

sample = pd.DataFrame(json.load(open(SAMPLE_PATH, encoding="utf-8")))
test   = pd.read_csv(TEST_PATH)
sub    = pd.read_csv(SUB_PATH)

if "id" not in test.columns:
    test.insert(0, "id", range(len(test)))
    print("⚠ Added synthetic id column to test")
for df in (sample,test):
    df["no_ctx"]=df["context"].map(is_no_ctx)
    df["ctx_clean"]=df.apply(lambda r:"" if r["no_ctx"] else str(r["context"]),axis=1)
    df["premise"]=df.apply(lambda r: str(r["prompt_bn"]) if r["no_ctx"]
                           else (str(r["prompt_bn"])+" "+r["ctx_clean"]).strip(),axis=1)
    df["response"]=df["response_bn"].astype(str)
assert list(sub.columns)==["id","label"] and len(sub)==len(test)
print("val:",sample.shape,"| halluc:",round((sample.label==0).mean(),3),
      "| test has/no ctx:",int((~test.no_ctx).sum()),int(test.no_ctx.sum()))

# --- LEAKAGE AUDIT (Stage 0 of the architecture) ---
# exact (prompt, response) matches between the labeled sample and the test set
# carry a KNOWN label — the submission cell copies it over the model prediction.
def _leak_key(p, r):
    return re.sub(r"\s+", " ", str(p).strip().lower()) + " || " + re.sub(r"\s+", " ", str(r).strip().lower())
_known = {_leak_key(p, r): int(l)
          for p, r, l in zip(sample["prompt_bn"], sample["response_bn"], sample["label"])}
test["leak_label"] = [_known.get(_leak_key(p, r), np.nan)
                      for p, r in zip(test["prompt_bn"], test["response_bn"])]
print("leakage audit | duplicate test ids:", int(test["id"].duplicated().sum()),
      "| exact sample->test matches:", int(test["leak_label"].notna().sum()))




```

```python
# ===== CELL 5 — NLI SOURCES (TSV + IndicXNLI-bn; token optional, local-first) =====
def load_nli_tsv():
    if not os.path.exists(cfg.nli_tsv):
        print("tsv missing"); return pd.DataFrame()

    raw = pd.read_csv(cfg.nli_tsv, sep="\t", on_bad_lines="skip").dropna()
    rows = []

    for _, r in raw.iterrows():
        p = str(r.get("Premise", "")).strip()
        e = str(r.get("Entailment", "")).strip()
        c = str(r.get("Contradiction", "")).strip()
        n = str(r.get("Neutral", "")).strip()

        if p and e: rows.append((p, e, 1, "nli"))
        if p and c: rows.append((p, c, 0, "nli"))
        # Neutral dropped # Neutral treated as Hallucination (0)

    return pd.DataFrame(rows, columns=["premise", "response", "label", "src"]).dropna()

def load_indicxnli():
    try:
        from datasets import load_dataset
        d = load_dataset(cfg.hf_ixnli, "bn", split="train", token=HF_TOKEN)
        df = pd.DataFrame({"premise":d["premise"], "response":d["hypothesis"], "label":d["label"]})
        df["label"] = (df["label"]==0).astype(int)  # XNLI: 0=entail,1=neutral,2=contra
        df["src"] = "ixnli"
        return df.sample(min(15000,len(df)), random_state=SEED)
    except Exception as e:
        print("IndicXNLI skipped:", str(e)[:100])
        return pd.DataFrame()

nli_df = pd.concat([load_nli_tsv(), load_indicxnli()], ignore_index=True)
print("NLI total:", nli_df.shape, nli_df["label"].value_counts().to_dict() if len(nli_df) else {})




```

```python
# ===== CELL 6 — REAL BENGALI QA & BANGLA HALLU EVAL =====
def qa_rows(items):
    by_ctx={}
    for it in items: by_ctx.setdefault(it.get("context",""),[]).append(it)
    all_ans=[str(it["answer"]) for it in items if it.get("answer")]
    rows=[]
    for ctx,grp in by_ctx.items():
        ans_list = list({str(g["answer"]) for g in grp if g.get("answer")})
        for g in grp:
            q=str(g["question"]); prem=(q+" "+str(ctx)).strip()
            if g.get("answer"):
                a=str(g["answer"]); rows.append((prem,a,1,"faithful"))
                others=[x for x in ans_list if x!=a]
                if others: rows.append((prem,random.choice(others),0,"wrong_attr"))
                if DIGIT_RE.search(a):
                    c=bump_digits(a)
                    if c: rows.append((prem,c,0,"intrinsic"))
                else:
                    for _ in range(6):
                        cand=str(random.choice(all_ans))
                        if cand!=a and cand not in ctx:
                            rows.append((prem,cand,0,"extrinsic")); break
            else:
                w=[t for t in str(ctx).split() if len(t)>2]
                if w:
                    i=random.randrange(max(1,len(w)-2))
                    rows.append((prem," ".join(w[i:i+2]),0,"unanswerable"))
    return pd.DataFrame(rows,columns=["premise","response","label","mode"])

def load_squad_bn():
    try:
        from datasets import load_dataset
        d=load_dataset(cfg.hf_squad,split="train",token=HF_TOKEN)
        return [{"context":r["context"],"question":r["question"],"answer":r["answers"]["text"][0] if r["answers"]["text"] else None} for r in d]
    except: return []
    
def load_qa_70k():
    try:
        from datasets import load_dataset
        d=load_dataset(cfg.hf_qa_70k,split="train",token=HF_TOKEN)
        return [{"context": "", "question": str(r.get("question", "")).strip(), "answer": str(r.get("answer", "")).strip()} for r in d if r.get("question") and r.get("answer")]
    except: return []

# --- DYNAMIC LOADER FOR BHE DATASETS ---
def load_bhe_datasets():
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

qa_items=load_squad_bn()+load_qa_70k()
random.shuffle(qa_items); qa_items=qa_items[:10000]
qa_df=qa_rows(qa_items) if qa_items else pd.DataFrame(columns=["premise","response","label","mode"])
if len(qa_df): qa_df["src"]="qa"

bhe_df = load_bhe_datasets()
if len(qa_df) and len(bhe_df):
    qa_df = pd.concat([qa_df, bhe_df], ignore_index=True)
elif len(bhe_df):
    qa_df = bhe_df

print("QA + BHE total:",qa_df.shape,"| modes:",qa_df["mode"].value_counts().to_dict() if len(qa_df) else {})




```

```python
# ===== CELL 7 — CLOZE SYNTHETIC FROM WIKI =====
STOP=set("এবং ও কিন্তু বা যে যা এই সেই তার তাদের করা হয় হয়ে হন ছিল ছিলেন একটি একটা এক থেকে সালে জন্য এর কে না নয় করে করেন দিয়ে পরে আগে মধ্যে সাথে হিসেবে".split())
def spans_in(s):
    out=[m.group() for m in DIGIT_RE.finditer(s)]; w=toks(s)
    for n in (1,2,3):
        for i in range(len(w)-n+1):
            g=w[i:i+n]
            if g[0] in STOP or g[-1] in STOP: continue
            if all(len(x)<3 for x in g): continue
            if any(DIGIT_RE.fullmatch(x) for x in g): continue
            sp=" ".join(g)
            if mostly_bengali(sp): out.append(sp)
    return list(dict.fromkeys(out))
def load_wiki(limit):
    files=[]
    for s in ("train/train","valid/valid",""): files+=glob.glob(os.path.join(cfg.wiki_dir,s,"*.txt"))
    if hasattr(cfg, "books_dir") and type(cfg.books_dir) == str and len(cfg.books_dir) > 0: files+=glob.glob(os.path.join(cfg.books_dir, "**/*.txt"), recursive=True)
    random.shuffle(files); out=[]
    for fp in files[:limit]:
        try:
            t=open(fp,encoding="utf-8",errors="ignore").read()
            if len(t)>200 and mostly_bengali(t[:400]): out.append(t[:3000])
        except: pass
    return out
def build_cloze(passages):
    pp=[]
    for p in passages:
        sp=[]
        for s in sent_split(p): sp+=[(s,x) for x in spans_in(s)]
        pp.append(sp)
    rows=[]
    for pi,p in enumerate(passages):
        cand=pp[pi]
        if len(cand)<4: continue
        numc=[(s,x) for s,x in cand if DIGIT_RE.fullmatch(x)]
        sent,span=random.choice(numc) if (numc and random.random()<0.5) else random.choice(cand)
        if span not in sent: continue
        prem=sent.replace(span,"____",1)+" — শূন্যস্থানে কী বসবে? "+p
        rows.append((prem,span,1,"faithful"))
        if DIGIT_RE.fullmatch(span):
            c=bump_digits(span)
            if c: rows.append((prem,c,0,"intrinsic"))
        others=[x for s2,x in cand if x!=span and x not in sent]
        if others: rows.append((prem,random.choice(others),0,"wrong_attr"))
        oj=random.randrange(len(passages))
        if oj!=pi and pp[oj]: rows.append((prem,random.choice(pp[oj])[1],0,"extrinsic"))
    df=pd.DataFrame(rows,columns=["premise","response","label","mode"]); df["src"]="synth"; return df
wiki_passages=load_wiki(cfg.n_wiki_files); print("wiki passages:",len(wiki_passages))
synth_df=build_cloze(wiki_passages) if wiki_passages else pd.DataFrame(columns=["premise","response","label","mode","src"])
print("cloze synthetic:",synth_df.shape)




```

```python
# ===== CELL 8 — ASSEMBLE + MODE-STRATIFIED 50/50 BALANCE =====
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
MAX_RATIO = 2.0
k0 = min(len(c0), int(len(c1) * MAX_RATIO))
k1 = min(len(c1), int(len(c0) * MAX_RATIO))
if len(c0)>k0:
    c0=(c0.groupby("mode",group_keys=False)
          .apply(lambda g:g.sample(max(1,int(round(k0*len(g)/len(train_all[train_all.label==0])))),random_state=SEED)))
    c0=c0.sample(min(len(c0),k0),random_state=SEED)
if len(c1)>k1: c1=c1.sample(k1,random_state=SEED)
train_all=pd.concat([c1,c0]).sample(frac=1,random_state=SEED).reset_index(drop=True)

n_hold=min(3000,len(train_all)//10)
synth_hold=train_all.iloc[:n_hold].reset_index(drop=True)
train_main=train_all.iloc[n_hold:].reset_index(drop=True)
print("train:",train_main.shape,"| labels:",train_main.label.value_counts().to_dict())




```

```python
# ===== CELL 9 — DATASET · FOCAL · TRAIN/PREDICT (fp32 params + fp16 autocast) =====
class PairDS(Dataset):
    def __init__(self,df,tok,mx,lab=True):
        self.p=df["premise"].astype(str).tolist(); self.h=df["response"].astype(str).tolist()
        self.y=df["label"].tolist() if lab else None; self.t=tok; self.m=mx
    def __len__(self): return len(self.p)
    def __getitem__(self,i):
        e=self.t(self.p[i],self.h[i],truncation=True,max_length=self.m,padding="max_length",return_tensors="pt")
        it={"input_ids":e["input_ids"].squeeze(0),"attention_mask":e["attention_mask"].squeeze(0)}
        if self.y is not None: it["labels"]=torch.tensor(self.y[i],dtype=torch.long)
        return it
class Focal(nn.Module):
    def __init__(self,gamma,alpha):
        super().__init__(); self.g=gamma; self.register_buffer("w",torch.tensor([alpha,1.0]))
    def forward(self,lg,y):
        ce=F.cross_entropy(lg,y,weight=self.w.to(lg.device),reduction="none")
        pt=torch.exp(-ce); return ((1-pt)**self.g*ce).mean()
def resolve_model(name,hf):
    for c in (f"/kaggle/input/{name}",f"/kaggle/input/{name}/{name}"):
        if os.path.exists(os.path.join(c,"config.json")): return c
    return hf
@torch.no_grad()
def predict_proba(model,tok,df,mx,bs):
    model.eval(); out=[]
    for b in DataLoader(PairDS(df,tok,mx,lab=False),batch_size=bs,shuffle=False):
        with torch.amp.autocast("cuda",dtype=torch.float16):
            lg=model(input_ids=b["input_ids"].to(DEVICE),attention_mask=b["attention_mask"].to(DEVICE)).logits.float()
        out.append(torch.softmax(lg,-1)[:,1].cpu().numpy())
    return np.concatenate(out)
def train_backbone(name,hf,tr,val,seed=SEED,quiet=False):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    path=resolve_model(name,hf); tok=AutoTokenizer.from_pretrained(path)
    model=AutoModelForSequenceClassification.from_pretrained(
        path,num_labels=2,ignore_mismatched_sizes=True).float().to(DEVICE)
    crit=Focal(cfg.focal_gamma,cfg.focal_alpha)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=0.01)
    ld=DataLoader(PairDS(tr.sample(frac=1,random_state=seed),tok,cfg.max_len),
                  batch_size=cfg.batch_size,shuffle=True,pin_memory=True)
    tot=len(ld)*cfg.epochs; sch=get_cosine_schedule_with_warmup(opt,int(tot*cfg.warmup),tot)
    scaler=torch.amp.GradScaler("cuda")
    for ep in range(cfg.epochs):
        model.train()
        for b in ld:
            opt.zero_grad()
            with torch.amp.autocast("cuda",dtype=torch.float16):
                lg=model(input_ids=b["input_ids"].to(DEVICE),attention_mask=b["attention_mask"].to(DEVICE)).logits
                loss=crit(lg,b["labels"].to(DEVICE))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        f=f1_score(val["label"],(predict_proba(model,tok,val,cfg.max_len,cfg.batch_size*2)>=0.5).astype(int),pos_label=0)
        if not quiet: print(f"  {name} s{seed} ep{ep+1}: valF1(c0)={f:.4f}")
    del opt, scaler, ld, crit
    import gc; gc.collect(); torch.cuda.empty_cache()
    return model, tok




```

```python
# ===== CELL 10 — ENCODER ENSEMBLE (5-FOLD MULTI-MODEL OUT-OF-FOLD) =====
sig_val = {}; sig_test = {}; keep_for_retr = None
import os, gc, glob, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# The 3 backbones driving the semantic net
cfg.backbones = (
    ("banglabert_large", "csebuetnlp/banglabert_large"),
    ("mdeberta", "microsoft/mdeberta-v3-base"),
    ("xlm_roberta", "xlm-roberta-large")
)

def find_fold_ckpt(key, fold_idx):
    patterns = [
        f"/kaggle/input/**/{key}_fold{fold_idx}.pt",
        f"/kaggle/input/**/{key}_fold{fold_idx}/{key}_fold{fold_idx}.pt",
        f"/kaggle/input/**/{key}.pt" # Absolute baseline fallback
    ]
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits: return hits[0]
    return None

first_key = cfg.backbones[0][0]
enc_keys = []

for bb_key, bb_path in cfg.backbones:
    fold_preds_val = []
    fold_preds_test = []
    
    for fold in range(5):
        ckpt = find_fold_ckpt(bb_key, fold)
        if not ckpt:
            continue
            
        print(f"[{bb_key}] Loading fold framework {fold} from {ckpt}")
        tk = AutoTokenizer.from_pretrained(bb_path)
        m = AutoModelForSequenceClassification.from_pretrained(
            bb_path, num_labels=2, ignore_mismatched_sizes=True).float().to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        m.eval()
        
        fold_preds_val.append(predict_proba(m, tk, sample, cfg.max_len, cfg.batch_size*2))
        fold_preds_test.append(predict_proba(m, tk, test, cfg.max_len, cfg.batch_size*2))
        
        # Save a single frozen slice of fold 0 for FAISS matching matrix later
        if cfg.use_retrieval and bb_key == first_key and fold == 0:
            keep_for_retr = (m.half().eval(), tk)
        else:
            m = m.cpu(); del m; gc.collect(); torch.cuda.empty_cache()
            
    if fold_preds_val:
        sig_val[bb_key] = np.mean(fold_preds_val, axis=0)
        sig_test[bb_key] = np.mean(fold_preds_test, axis=0)
        enc_keys.append(bb_key)
        print(f"✨ [{bb_key} Combined 5-Fold] OOF F1 evaluation computed.")

# --- Combine features down to a single meta signal entry ---
sig_val["enc"]  = np.mean([sig_val[k]  for k in enc_keys], axis=0)
sig_test["enc"] = np.mean([sig_test[k] for k in enc_keys], axis=0)
print(f"🔥 [FINAL ENCODER META-SIGNAL] val F1(c0)@0.5 = {f1_score(sample['label'],(sig_val['enc']>=0.5).astype(int),pos_label=0):.4f}")
tleft()




```

```python
# ===== CELL 11 — RETRIEVAL-AUGMENTED no_context (`retr`) — FAISS + Dense Embeddings =====
retr_sim_val = np.full(len(sample), np.nan)
retr_sim_test = np.full(len(test), np.nan)

def build_retr_signal():
    global retr_sim_val, retr_sim_test
    keep_for_retr_val = globals().get("keep_for_retr")
    if not (cfg.use_retrieval and wiki_passages and keep_for_retr_val):
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

    model, tok = keep_for_retr_val

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
if globals().get("keep_for_retr"):
    del globals()["keep_for_retr"]
    gc.collect()
    torch.cuda.empty_cache()




```

```python
# ===== CELL 12 — LEX/NUM (has_context) =====
def lexnum(df):
    p=np.array([contain(r["response_bn"],r["ctx_clean"]) for _,r in df.iterrows()])
    nu=np.array([len(numset(r["response_bn"])-numset(r["ctx_clean"])) for _,r in df.iterrows()])
    s=0.7*p+0.3*(nu==0); s[df["no_ctx"].values]=np.nan; return s
lex_val=lexnum(sample); lex_test=lexnum(test)




```

```python
# ===== CELL 12.5 — NUCLEAR CLEAR & MEMORY BUDGET =====
import gc, torch, ctypes

def nuclear_clear():
    suspects = ['m','model','tk','tok','tokenizer','keep_for_retr',
                'opt','sch','scaler','ld','crit','backbone']
    for name in suspects:
        if name in globals(): del globals()[name]
    
    for _ in range(3): gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except: pass
    
    if torch.cuda.is_available():
        total_free = 0
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            total_free += free
            alloc = torch.cuda.memory_allocated(i)/1e9
            print(f"GPU{i}: {free/1e9:.1f}GB free / {total/1e9:.1f}GB | alloc={alloc:.1f}GB")
        
        free0 = torch.cuda.mem_get_info(0)[0]
        free1 = torch.cuda.mem_get_info(1)[0]
        cfg.max_mem_llm = {
            0: f"{max(1, int(free0/1e9*0.75))}GiB",
            1: f"{max(1, int(free1/1e9*0.85))}GiB",
            "cpu": "40GiB"
        }
        print(f"LLM budget: {cfg.max_mem_llm}")
        
        if total_free < 8e9:
            print("⚠ Less than 8GB free total — LLM may OOM")
        else:
            print("✅ GPU clear — safe to load LLM")

nuclear_clear()





```

```python
# ===== CELL 13 — DUAL-LLM JUDGE (SEQUENTIAL, SELF-CONTAINED) =====
cfg.llm_input_len = 512
import re, os, gc, torch, ctypes, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import f1_score

# ── Helper: detect math/logic questions ──────────────────────────────────────
def is_math_or_logic(prompt, ctx=""):
    math_terms = ["কত", "যোগ", "বিয়োগ", "গুণ", "ভাগ", "শতকরা",
                  "শতাংশ", "গণিত", "হিসাব", "সংখ্যা"]
    if any(m in str(prompt) for m in math_terms): return True
    if re.search(r"\d+", str(prompt)): return True
    return False

def is_no_ctx(ctx):
    return not ctx or str(ctx).strip().lower() in {"", "null", "none", "nan", "[null]"}

# ── Helper: route each row to a category ─────────────────────────────────────
def get_category(prompt_text, ctx_text, response_text):
    p_lower   = str(prompt_text).lower()
    combined  = f"{prompt_text} {ctx_text} {response_text}"
    if re.search(r"[a-zA-Z]", combined):
        return "code_mixed"
    elif ctx_text and not is_no_ctx(ctx_text):
        return "comprehension"
    elif any(k in p_lower for k in ["অর্থ", "ভাবার্থ", "সমার্থক",
                                     "বিপরীত", "মানে কী"]):
        return "vocabulary"
    elif is_math_or_logic(prompt_text):
        return "math"
    else:
        return "general_knowledge"

# ── Helper: build system prompt per category ─────────────────────────────────
def build_sys_prompt(category):
    base = "কোনো ব্যাখ্যা দেবেন না। শুধুমাত্র একটি সংখ্যা আউটপুট দিন।"
    prompts = {
        "code_mixed": (
            "আপনি একজন বহুভাষিক ডেটা বিশ্লেষক। ভাষার মিশ্রণ থাকা সত্ত্বেও "
            "মূল অর্থটি যাচাই করুন। উত্তর সঠিক হলে '1', ভুল বা বানোয়াট হলে '0' লিখুন। "
        ),
        "comprehension": (
            "আপনি একটি হ্যালুসিনেশন সনাক্তকরণ এআই। অনুচ্ছেদের তথ্যের সাথে "
            "উত্তর মিললে '1', সাংঘর্ষিক হলে '0' লিখুন। "
        ),
        "vocabulary": (
            "আপনি একজন বাংলা ভাষাবিদ। শব্দের সঠিক অর্থ হলে '1', "
            "ভুল বা মনগড়া হলে '0' লিখুন। "
        ),
        "math": (
            "আপনি একজন গাণিতিক মূল্যায়নকারী। হিসাব সম্পূর্ণ সঠিক হলে '1', "
            "সামান্যতম ভুল থাকলে '0' লিখুন। "
        ),
        "general_knowledge": (
            "আপনি একজন তথ্য-যাচাইকারী। উত্তর সম্পূর্ণ সত্য হলে '1', "
            "ভুল বা বানোয়াট হলে '0' লিখুন। "
        ),
    }
    return prompts.get(category, prompts["general_knowledge"]) + base

# ── Helper: get token IDs for 0 and 1 ────────────────────────────────────────
def digit_ids(tk, d):
    ids = set()
    for s in (d, f" {d}", f"▁{d}"):
        e = tk.encode(s, add_special_tokens=False)
        if e: ids.add(e[-1])
    return list(ids) if ids else [tk.encode(d, add_special_tokens=False)[-1]]

# ── Core: score one dataframe with one LLM ───────────────────────────────────
def score_df_with_llm(df, llm, tk, ids1, ids0, dev, model_title):
    out = np.zeros(len(df))
    for i, r in enumerate(df.itertuples()):
        ctx = getattr(r, "ctx_clean", "")
        if is_no_ctx(ctx): ctx = ""
        cat = get_category(r.prompt_bn, ctx, r.response_bn)
        SYS = build_sys_prompt(cat)
        u   = (f"CONTEXT: {ctx}\n" if ctx else "") + \
              f"QUESTION: {r.prompt_bn}\nANSWER: {r.response_bn}\nVerdict:"
        enc = tk.apply_chat_template(
            [{"role": "system", "content": SYS},
             {"role": "user",   "content": u}],
            add_generation_prompt=True,
            return_tensors="pt", return_dict=True)
        ii = enc["input_ids"][:,  -cfg.llm_input_len:].to(dev)
        am = enc["attention_mask"][:, -cfg.llm_input_len:].to(dev)
        with torch.no_grad():
            lg = llm(input_ids=ii, attention_mask=am).logits[0, -1, :].float()
        p1 = torch.logsumexp(lg[ids1], 0)
        p0 = torch.logsumexp(lg[ids0], 0)
        out[i] = torch.softmax(torch.stack([p0, p1]), 0)[1].item()
        if i % 200 == 0:
            print(f"  [{model_title}] {i}/{len(df)} | cat={cat}")
    return out

# ── Nuclear memory clear ──────────────────────────────────────────────────────
def nuke_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except: pass
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU{i}: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total")

# ── Main engine: load one LLM, score val+test, delete ────────────────────────
def run_llm_subengine(df_val, df_test, model_title, path_or_id, use_prequant=False):
    if not cfg.use_llm_judge:
        return np.full(len(df_val), np.nan), np.full(len(df_test), np.nan)

    print(f"\n{'='*60}")
    print(f"🚀 Loading: {model_title}")
    nuke_gpu()

    # Dynamic memory budget
    free0 = torch.cuda.mem_get_info(0)[0] if torch.cuda.is_available() else 0
    free1 = torch.cuda.mem_get_info(1)[0] if torch.cuda.device_count() > 1 else 0
    max_mem = {
        0: f"{max(1, int(free0 / 1e9 * 0.80))}GiB",
        1: f"{max(1, int(free1 / 1e9 * 0.85))}GiB",
        "cpu": "40GiB",
    }
    print(f"  Memory budget: {max_mem}")

    tk  = AutoTokenizer.from_pretrained(path_or_id)
    llm = None

    if use_prequant:
        try:
            llm = AutoModelForCausalLM.from_pretrained(
                path_or_id,
                device_map="balanced",
                max_memory=max_mem,
            ).eval()
            print(f"  ✅ Loaded pre-quantized {model_title}")
        except RuntimeError as e:
            if "memory" not in str(e).lower(): raise
            print(f"  ⚠ OOM on balanced load — trying cpu offload")
            nuke_gpu()
            max_mem_offload = {0: "6GiB", 1: "8GiB", "cpu": "40GiB"}
            llm = AutoModelForCausalLM.from_pretrained(
                path_or_id,
                device_map="auto",
                max_memory=max_mem_offload,
            ).eval()
            print(f"  ✅ Loaded with CPU offload")
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        try:
            llm = AutoModelForCausalLM.from_pretrained(
                path_or_id,
                quantization_config=bnb,
                device_map="balanced",
                max_memory=max_mem,
            ).eval()
            print(f"  ✅ Loaded 4-bit {model_title}")
        except RuntimeError as e:
            if "memory" not in str(e).lower(): raise
            print(f"  ⚠ OOM — skipping {model_title}")
            del tk; gc.collect(); torch.cuda.empty_cache()
            return np.full(len(df_val), np.nan), np.full(len(df_test), np.nan)

    dev  = next(llm.parameters()).device
    ids1 = digit_ids(tk, "1")
    ids0 = digit_ids(tk, "0")
    print(f"  Primary device: {dev} | ids1={ids1} | ids0={ids0}")

    print(f"  Scoring val ({len(df_val)} rows)...")
    val_out  = score_df_with_llm(df_val,  llm, tk, ids1, ids0, dev, model_title)

    print(f"  Scoring test ({len(df_test)} rows)...")
    test_out = score_df_with_llm(df_test, llm, tk, ids1, ids0, dev, model_title)

    del llm, tk
    nuke_gpu()
    return val_out, test_out

# ══════════════════════════════════════════════════════════════════════════════
# SEGMENT A — TigerLLM-9B-it (pre-quantized, Bengali-native, ~7GB)
# ══════════════════════════════════════════════════════════════════════════════
PREQUANT_PATH = "/kaggle/input/datasets/bayazidhs/tigerllm-9b-4bit/tigerllm-9b-4bit"

tiger_val, tiger_test = run_llm_subengine(
    sample, test,
    model_title   = "TigerLLM-9B-it [Pre-quantized]",
    path_or_id    = PREQUANT_PATH,
    use_prequant  = True,
)
print(f"  TigerLLM val F1(c0)@0.5 = "
      f"{f1_score(sample[label], (tiger_val>=0.5).astype(int), pos_label=0):.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# SEGMENT B — Qwen2.5-3B-Instruct (multilingual reasoning, ~2GB quantized)
# ══════════════════════════════════════════════════════════════════════════════
qwen_val, qwen_test = run_llm_subengine(
    sample, test,
    model_title  = "Qwen2.5-3B-Instruct",
    path_or_id   = "Qwen/Qwen2.5-3B-Instruct",
    use_prequant = False,
)
print(f"  Qwen val F1(c0)@0.5 = "
      f"{f1_score(sample[label], (qwen_val>=0.5).astype(int), pos_label=0):.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# COMBINE — weighted average (TigerLLM=0.6, Qwen=0.4)
# ══════════════════════════════════════════════════════════════════════════════
def safe_blend(a, b, w_a=0.6, w_b=0.4):
    a_nan = np.isnan(a)
    b_nan = np.isnan(b)
    out   = np.full(len(a), np.nan)
    both  = ~a_nan & ~b_nan
    only_a = ~a_nan & b_nan
    only_b =  a_nan & ~b_nan
    out[both]   = w_a * a[both]   + w_b * b[both]
    out[only_a] = a[only_a]
    out[only_b] = b[only_b]
    return out

llm_val  = safe_blend(tiger_val,  qwen_val,  w_a=0.6, w_b=0.4)
llm_test = safe_blend(tiger_test, qwen_test, w_a=0.6, w_b=0.4)

llm_val  = np.where(np.isnan(llm_val),  0.5, llm_val)
llm_test = np.where(np.isnan(llm_test), 0.5, llm_test)

print(f"\n🔥 [DUAL-LLM COMBINED] val F1(c0)@0.5 = "
      f"{f1_score(sample[label], (llm_val>=0.5).astype(int), pos_label=0):.4f}")
print(f"  tiger NaN: {np.isnan(tiger_val).sum()} | "
      f"qwen NaN: {np.isnan(qwen_val).sum()} | "
      f"combined NaN: 0")

tleft()

```

```python
# ===== CELL 14 — RANK-NORMALIZE SIGNALS + META FEATURES (val∪test) =====
SIGNAL_COLS = ("enc", "lex", "retr", "llm")

def tfidf_prompt_ctx_sim(df, vec=None):
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
    return sim, vec

def stackX(df, sv, lex, retr, llm, retr_sim=None):
    X = pd.DataFrame()
    X["enc"] = sv["enc"] if "enc" in sv else np.nan
    X["lex"] = lex if lex is not None else np.nan
    X["retr"] = retr if retr is not None else np.nan
    X["llm"] = llm if llm is not None else np.nan
    X["no_ctx"] = df["no_ctx"].values
    return X


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

from scipy.stats import skew, kurtosis

def add_meta_features(df, X, retr_sim=None, tfidf_vec=None):
    X = X.copy()
    X["prompt_len"] = df["prompt_bn"].astype(str).str.len().values
    X["ctx_len"] = df["ctx_clean"].astype(str).str.len().values
    X["resp_len"] = df["response_bn"].astype(str).str.len().values
    sim_res, tfidf_vec_out = tfidf_prompt_ctx_sim(df, tfidf_vec)
    X["tfidf_sim"] = sim_res
    if retr_sim is not None:
        X["retr_sim"] = retr_sim

    CORE = ["enc", "lex", "retr", "llm"]
    sig_matrix = np.column_stack([
        X[c].fillna(0.5).values if c in X.columns else np.full(len(df), 0.5)
        for c in CORE
    ])
    X["signal_skew"] = skew(sig_matrix, axis=1, bias=True)
    X["signal_kurt"] = kurtosis(sig_matrix, axis=1, bias=True)
    X["signal_std"] = sig_matrix.std(axis=1)
    X["signal_range"] = sig_matrix.max(axis=1) - sig_matrix.min(axis=1)
    X["signal_max_mean_gap"] = sig_matrix.max(axis=1) - sig_matrix.mean(axis=1)
    X["n_signals_hallu"] = (sig_matrix < 0.5).sum(axis=1).astype(float)

    sig_raw = np.column_stack([
        X[c].values if c in X.columns else np.full(len(df), np.nan)
        for c in CORE
    ])
    X["n_signals_missing"] = np.isnan(sig_raw).sum(axis=1).astype(float)

    # Category encoding for LightGBM (C1 cultural-distance routing)
    CATEGORY_MAP = {"comprehension": 0, "math": 1, "vocabulary": 2,
                    "general_knowledge": 3, "history": 4, "code_mixed": 5}
    if "category" in df.columns:
        X["category_enc"] = df["category"].map(CATEGORY_MAP).fillna(3).values
    else:
        X["category_enc"] = 3.0  # default to general_knowledge
    # --- regime flags (Task/Regime Router -> meta-model) ---
    TRANSLATE_PAT = re.compile(r'অনুবাদ|translate|ইংরেজিতে|বাংলায়|রূপান্তর', re.I)
    SUMMARY_PAT   = re.compile(r'সারাংশ|সংক্ষেপে|summar', re.I)
    pr = df["prompt_bn"].astype(str)
    rs = df["response_bn"].astype(str)
    cx = df["ctx_clean"].astype(str)
    
    X["regime_context"]     = (~df["no_ctx"].values).astype(float)
    X["regime_factual"]     = (df["no_ctx"].values & ~pr.str.contains(TRANSLATE_PAT) & ~pr.str.contains(SUMMARY_PAT)).astype(float)
    X["regime_translation"] = pr.str.contains(TRANSLATE_PAT, regex=True, na=False).astype(float)
    X["regime_summary"]     = pr.str.contains(SUMMARY_PAT, regex=True, na=False).astype(float)
    
    X["is_math"] = [int(bool(numset(p)) and bool(numset(r))) for p, r in zip(pr, rs)]
    X["is_translation"] = pr.str.contains("অনুবাদ|translate|ইংরেজিতে|সারাংশ|সংক্ষেপে|summar", regex=True, case=False).astype(int).values
    X["is_mcq"] = (pr.str.contains(r"ক\)", regex=True) & pr.str.contains(r"খ\)", regex=True)).astype(int).values
    def _numsup(p, r, c):
        nr = numset(r)
        return -1.0 if not nr else len(nr & (numset(p) | numset(c))) / len(nr)
    X["number_support"] = [_numsup(p, r, c) for p, r, c in zip(pr, rs, cx)]
    return X
Xv = stackX(sample, sig_val, lex_val, retr_val, llm_val, retr_sim_val)
Xt = stackX(test, sig_test, lex_test, retr_test, llm_test, retr_sim_test)
Xv, fitted_tfidf = add_meta_features(sample, Xv, retr_sim_val, None)
Xt, _ = add_meta_features(test, Xt, retr_sim_test, fitted_tfidf)
Xv, Xt = z_score_norm(Xv, Xt)
yv = sample["label"].values
print("signals:", [c for c in Xv.columns if c != "no_ctx"])






```

```python
# ===== CELL 15 — LIGHTGBM META-MODEL STACKING (replaces Powell blender) =====
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

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
    Xr = X.loc[mask, FEAT_COLS].reset_index(drop=True)
    yr = y[mask]
    
    params = dict(
        objective="binary", metric="binary_logloss", verbosity=-1, seed=seed,
        learning_rate=0.02, num_leaves=7, min_data_in_leaf=15,
        feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
        lambda_l1=0.1, lambda_l2=0.5
    )
    
    # 5-Fold OOF Predictions for Threshold Tuning
    oof_p = np.zeros(len(yr))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    for trn_idx, val_idx in skf.split(Xr, yr):
        X_trn, y_trn = Xr.iloc[trn_idx], yr[trn_idx]
        X_val = Xr.iloc[val_idx]
        
        m_cv = lgb.train(params, lgb.Dataset(X_trn, label=y_trn, categorical_feature=['category_enc']), num_boost_round=35)
        oof_p[val_idx] = m_cv.predict(X_val)
        
    t = tune_threshold(oof_p, yr)
    oof_f1 = f1c0(yr, oof_p, t)
    
    # Final Model trained on 100% of data
    model = lgb.train(params, lgb.Dataset(Xr, label=yr, categorical_feature=['category_enc']), num_boost_round=35)
    return model, t, oof_f1, oof_p

def lgbm_predict(X, model):
    return model.predict(X[FEAT_COLS])

mask_ctx = ~sample["no_ctx"].values
mask_no = sample["no_ctx"].values
lgb_ctx, tc, fc, oof_ctx = fit_lgbm(Xv, yv, mask_ctx)
lgb_noctx, tn, fn, oof_no = fit_lgbm(Xv, yv, mask_no)

# honest validation view = OOF probabilities, never the refit model's in-sample fit
def calibrate_probs(oof_p, y):
    cal = LogisticRegression(C=1.0, max_iter=1000)
    cal.fit(oof_p.reshape(-1,1), y)
    return cal

def apply_calibration(cal, p):
    return cal.predict_proba(p.reshape(-1,1))[:,1]

cal_ctx   = calibrate_probs(oof_ctx, yv[mask_ctx])
cal_noctx = calibrate_probs(oof_no,  yv[mask_no])

oof_ctx_cal = apply_calibration(cal_ctx,   oof_ctx)
oof_no_cal  = apply_calibration(cal_noctx, oof_no)

pv = np.zeros(len(sample))
pv[mask_ctx] = oof_ctx_cal
pv[mask_no]  = oof_no_cal

pt = np.zeros(len(test))
pt_ctx_raw = lgbm_predict(Xt.loc[~test["no_ctx"].values], lgb_ctx)
pt_no_raw  = lgbm_predict(Xt.loc[test["no_ctx"].values], lgb_noctx)
pt[~test["no_ctx"].values] = apply_calibration(cal_ctx,   pt_ctx_raw)
pt[test["no_ctx"].values]  = apply_calibration(cal_noctx, pt_no_raw)

tc = tune_threshold(oof_ctx_cal, yv[mask_ctx])
tn = tune_threshold(oof_no_cal,  yv[mask_no])

tv = np.where(sample["no_ctx"].values, tn, tc)
tt = np.where(test["no_ctx"].values, tn, tc)

print("LGBM has_ctx thr", round(tc, 3), "OOF pointF1", round(fc, 4))
print("LGBM no_ctx  thr", round(tn, 3), "OOF pointF1", round(fn, 4))
print(
    "OVERALL OOF F1(c0):",
    round(f1_score(yv, (pv >= tv).astype(int), pos_label=0), 4),
    "| all-0 floor:",
    round(f1_score(yv, np.zeros(len(yv)), pos_label=0), 4),
)

imp_ctx = pd.Series(lgb_ctx.feature_importance(), index=FEAT_COLS).sort_values(ascending=False)
imp_no = pd.Series(lgb_noctx.feature_importance(), index=FEAT_COLS).sort_values(ascending=False)
print("top features has_ctx:", {k: int(v) for k, v in imp_ctx.head(5).items()})
print("top features no_ctx:", {k: int(v) for k, v in imp_no.head(5).items()})






```

```python
# ===== CELL 15.5 — PSEUDO LABEL RETRAIN =====
if cfg.pseudo_label_n > 0:
    conf_mask = (pt < (tt - 0.25)) | (pt > (tt + 0.25))
    pseudo_df = test[conf_mask].copy()
    pseudo_df["label"] = (pt[conf_mask] >= tt[conf_mask]).astype(int)
    pseudo_df["_conf_score"] = abs(pt[conf_mask] - tt[conf_mask])
    pseudo_df = pseudo_df.nlargest(min(cfg.pseudo_label_n, len(pseudo_df)), "_conf_score")
    pseudo_df["premise"]  = pseudo_df["premise"]
    pseudo_df["response"] = pseudo_df["response_bn"].astype(str)
    pseudo_df["src"]      = "test_set"
    pseudo_df["mode"]     = "pseudo"
    pseudo_df[["premise","response","label","src","mode"]].to_csv(
        "/kaggle/working/pseudo_labels.csv", index=False)
    print(f"Saved {len(pseudo_df)} pseudo labels")




```

```python
# ===== CELL 16 — SUBMISSION =====
# pt/tt from Cell 15 (LightGBM meta-model); refined in Cell 15.5 if pseudo-retrain ran
out=pd.DataFrame({"id":test["id"].values,"label":(pt>=tt).astype(int)})
# exact duplicates of labeled sample rows get their known label (leakage audit, Cell 4)
if "leak_label" in test.columns:
    _lm = test["leak_label"].notna().values
    if _lm.any():
        out.loc[_lm, "label"] = test.loc[_lm, "leak_label"].astype(int).values
        print(f"leak override: {int(_lm.sum())} exact-match rows set to known labels")
assert list(out.columns)==["id","label"] and len(out)
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
print(f"False negatives (pred=0, true=1): {((wrong.pred==0)&(wrong.label==1)).sum()}")

# ── Hard Post-Processing Rules ────────
import re
final_preds = (pt >= tt).astype(int)
for i, r in enumerate(test.itertuples()):
    resp = str(getattr(r, "response_bn", ""))
    prompt = str(getattr(r, "prompt_bn", ""))
    
    # Rule 1: Empty response is always hallucinated
    if not resp or resp.strip() == "" or resp.lower() in ["nan", "null"]:
        final_preds[i] = 0
        
    # Rule 2: If prompt asks for a number, and response has no numbers -> hallucinated
    math_terms = ["কত", "কয়টি", "কয়টি", "কবে", "সাল", "তারিখ"]
    if any(m in prompt for m in math_terms):
        if not re.search(r'\d+|[০-৯]+', resp):
            final_preds[i] = 0

sub = pd.DataFrame({"id": test["id"], "label": final_preds})
sub.to_csv("submission.csv", index=False)
print("Saved final submission.csv with Hard Rules applied!")





```

```python
# ===== CELL 17 — DIAGNOSTICS =====
for reg,mask in (("has_ctx",~sample["no_ctx"].values),("no_ctx",sample["no_ctx"].values)):
    pr=(pv[mask]>=tv[mask]).astype(int)
    print(f"{reg}: n={mask.sum()} valF1(c0)={f1_score(yv[mask],pr,pos_label=0):.4f} "
          f"pred-halluc={np.mean(pr==0):.2f} true-halluc={np.mean(yv[mask]==0):.2f}")
diag=pd.DataFrame({"regime":np.where(sample.no_ctx,"no","has"),"label":yv,"p":pv})
for c in [c for c in Xv.columns if c!="no_ctx"]: diag[c]=Xv[c].values
diag.to_csv("/kaggle/working/val_signals.csv",index=False)
diag_test=pd.DataFrame({"regime":np.where(test.no_ctx,"no","has"),"p":pt})
for c in [c for c in Xt.columns if c!="no_ctx"]: diag_test[c]=Xt[c].values
diag_test.to_csv("/kaggle/working/test_signals.csv",index=False)
print("saved test_signals.csv")
print("saved val_signals.csv")




```

```python
# ===== CELL 18 — INTERACTIVE ERROR ANALYSIS & VISUALIZATIONS =====
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from IPython.display import display

print("Generating Interactive Visualizations...")

try:
    # 1. Probability Distribution by Label (Requires 'wrong' DataFrame from Cell 17)
    if 'wrong' in globals():
        fig_prob = px.histogram(wrong, x="prob", color="label", nbins=40,
                                title="Probability Distribution of Errors",
                                labels={"prob": "Predicted Probability", "label": "True Label"},
                                color_discrete_sequence=["#EF553B", "#00CC96"])
        fig_prob.update_layout(bargap=0.1)
        fig_prob.show()

    # 2. LightGBM Feature Importance
    if 'imp_ctx' in globals() and 'imp_no' in globals():
        weights_df = pd.DataFrame({
            'Feature': imp_ctx.index.tolist(),
            'Has Context': imp_ctx.values,
            'No Context': imp_no.reindex(imp_ctx.index).fillna(0).values,
        })
        fig_weights = px.bar(weights_df, x='Feature', y=['Has Context', 'No Context'], barmode='group',
                             title="LightGBM Meta-Model Feature Importance",
                             color_discrete_sequence=["#636EFA", "#FFA15A"])
        fig_weights.show()

    # 3. Source Breakdown of Sample (Val Set)
    if 'sample' in globals() and 'src' in sample.columns:
        fig_src = px.pie(sample, names='src', title="Validation Set Distribution by Source", hole=0.4)
        fig_src.show()
        
    # 4. Hallucination vs Faithful Distribution
    if 'sample' in globals() and 'label' in sample.columns:
        lbl_map = {0: "Hallucinated (0)", 1: "Faithful (1)"}
        dist_df = sample['label'].map(lbl_map).value_counts().reset_index()
        dist_df.columns = ['Label', 'Count']
        fig_dist = px.bar(dist_df, x='Label', y='Count', title="Overall Validation Label Distribution", color='Label')
        fig_dist.show()
except Exception as e:
    print(f"Visualization error: {e}")





```

