### As I have less runtime so Im using google collab

```python
import os

# 1. Inject your Kaggle credentials directly into the Colab environment
# (Replace these with your actual Kaggle username and text key from https://www.kaggle.com/settings/account)
os.environ['KAGGLE_USERNAME'] = "bayazidhs"
os.environ['KAGGLE_KEY'] = "KGAT_063be4398e520585086af3fbb9e5141f" # <--- REPLACE WITH YOUR ACTUAL KAGGLE KEY

# 2. Install and upgrade the Kaggle API
!pip install --upgrade -q kaggle

# 3. Create the necessary directory structure in a writable location
# The /kaggle/input directory in Colab is read-only, so we'll use /content/kaggle/input
!mkdir -p /content/kaggle/input/datasets/bayazidhs/bengali-hallucination-data
!mkdir -p /content/kaggle/input/datasets/ajmainmahtab/bangla-natural-language-inference-dataset
!mkdir -p /content/kaggle/input/datasets/disisbig/bengali-wikipedia-articles
!mkdir -p /content/kaggle/input/datasets/mahdihasanqurishi/banglahallueval-qa
!mkdir -p /content/kaggle/input/bangla-wikipedia-dataset

# 4. Download and Unzip the 5 Required Datasets
print("Downloading Host Data...")
!kaggle datasets download bayazidhs/bengali-hallucination-data -p /content/kaggle/input/datasets/bayazidhs/bengali-hallucination-data --unzip

print("Downloading NLI Dataset...")
!kaggle datasets download ajmainmahtab/bangla-natural-language-inference-dataset -p /content/kaggle/input/datasets/ajmainmahtab/bangla-natural-language-inference-dataset --unzip

print("Downloading Bengali Wikipedia Articles...")
!kaggle datasets download disisbig/bengali-wikipedia-articles -p /content/kaggle/input/datasets/disisbig/bengali-wikipedia-articles --unzip

print("Downloading BanglaHalluEval QA...")
!kaggle datasets download mahdihasanqurishi/banglahallueval-qa -p /content/kaggle/input/datasets/mahdihasanqurishi/banglahallueval-qa --unzip

print("Downloading Massive Hurutta Wikipedia Dump...")
!kaggle datasets download hurutta/bangla-wikipedia-dataset -p /content/kaggle/input/bangla-wikipedia-dataset --unzip

print("✅ Setup Complete without a JSON file!")





```

```python
# ==============================================================================
# COLAB ASSEMBLY & MULTI-FOLD TRAINING CELL
# ==============================================================================
import os, re, gc, glob, json, random, unicodedata, time, subprocess, sys
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

# Ensure working output folder exists
os.makedirs("/content/kaggle/working", exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Dynamic Path Resolution (Prevents folder structure mismatch)
def find_file(pattern):
    hits = glob.glob(pattern, recursive=True)
    return hits[0] if hits else None

SAMPLE_PATH = find_file("/content/**/dataset samples.json")
NLI_TSV     = find_file("/content/**/NLI Dataset - Combined.tsv")
WIKI_DIR    = os.path.dirname(find_file("/content/**/*.txt")) if find_file("/content/**/*.txt") else None
BHE_1000    = find_file("/content/**/banglahallueval_qa_1000.csv")
BHE_FULL    = find_file("/content/**/banglahallueval_qa_dataset.csv")
PSEUDO_PATH = find_file("/content/**/pseudo_labels.csv")

print(f"✓ Found Sample Path: {SAMPLE_PATH}")
print(f"✓ Found Wiki Dir: {WIKI_DIR}")

# 2. Text Normalization Utilities
BN_DIGITS = "০১২৩৪৫৬৭৮৯"; BN2ASCII = {ord(b):str(i) for i,b in enumerate(BN_DIGITS)}
BN_CHAR = re.compile(r"[\u0980-\u09FF]"); DIGIT_RE = re.compile(r"[০-৯0-9]+")
def norm(s): return unicodedata.normalize("NFC", str(s))
def denum(s): return norm(s).translate(BN2ASCII)
def mostly_bengali(s):
    s = str(s); b = len(BN_CHAR.findall(s))
    return b >= max(1, int(0.5 * len(re.findall(r"\S", s))))
def bump_digits(a):
    def b(ch):
        if ch in BN_DIGITS: return BN_DIGITS[(BN_DIGITS.index(ch)+random.randint(1,8))%10]
        if ch.isdigit(): return str((int(ch)+random.randint(1,8))%10)
        return ch
    n = "".join(b(c) for c in a); return None if n==a else n

# 3. Load & Structure Base Competition Rows
sample = pd.DataFrame(json.load(open(SAMPLE_PATH, encoding="utf-8")))
sample["no_ctx"] = sample["context"].map(lambda c: c is None or str(c).strip().lower() in {"", "null", "none", "nan"})
sample["ctx_clean"] = sample.apply(lambda r: "" if r["no_ctx"] else str(r["context"]), axis=1)
sample["premise"] = sample.apply(lambda r: str(r["prompt_bn"]) if r["no_ctx"] else (str(r["prompt_bn"])+" "+r["ctx_clean"]).strip(), axis=1)
sample["response"] = sample["response_bn"].astype(str)
sample["src"] = "host"

# 4. Process and Augment BanglaHalluEval (BHE) Data
bhe_rows = []
if BHE_1000 and os.path.exists(BHE_1000):
    df_bhe = pd.read_csv(BHE_1000)
    all_ans = df_bhe["correct_answer"].dropna().astype(str).tolist()
    for _, r in df_bhe.iterrows():
        q = str(r.get("question", "")).strip()
        ans = str(r.get("correct_answer", "")).strip()
        if not q or not ans: continue
        bhe_rows.append((q, ans, 1, "bhe"))
        if all_ans: bhe_rows.append((q, random.choice(all_ans), 0, "bhe"))

pseudo_rows = []
if PSEUDO_PATH and os.path.exists(PSEUDO_PATH):
    df_pseudo = pd.read_csv(PSEUDO_PATH)
    if {"premise", "response", "label"}.issubset(df_pseudo.columns):
        df_pseudo["src"] = "pseudo"
        pseudo_rows.append(df_pseudo[["premise", "response", "label", "src"]])
        print(f"✓ Injected {len(df_pseudo)} Pseudo-Labels into training!")

train_master = pd.concat([
    sample[["premise", "response", "label", "src"]], 
    pd.DataFrame(bhe_rows, columns=["premise", "response", "label", "src"])
] + pseudo_rows).dropna()
train_master = train_master.drop_duplicates(subset=["premise", "response"]).sample(frac=1, random_state=SEED).reset_index(drop=True)

# Cap size slightly to protect Colab memory limits
train_master = train_master.head(20000)
print(f"Master training matrix fully assembled. Rows: {len(train_master)}")

# 5. Training Dataset Structure & Loop Definition
class Focal(nn.Module):
    def __init__(self,gamma=2.0,alpha=0.75):
        super().__init__(); self.g=gamma; self.register_buffer("w",torch.tensor([alpha,1.0]))
    def forward(self,lg,y):
        ce=F.cross_entropy(lg,y,weight=self.w.to(lg.device),reduction="none")
        pt=torch.exp(-ce); return ((1-pt)**self.g*ce).mean()

class PairDS(Dataset):
    def __init__(self, df, tok, mx):
        self.p = df["premise"].astype(str).tolist(); self.h = df["response"].astype(str).tolist()
        self.y = df["label"].tolist(); self.t = tok; self.m = mx
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        e = self.t(self.p[i], self.h[i], truncation=True, max_length=self.m, padding="max_length", return_tensors="pt")
        return {"input_ids": e["input_ids"].squeeze(0), "attention_mask": e["attention_mask"].squeeze(0), "labels": torch.tensor(self.y[i], dtype=torch.long)}

def get_llrd_params(model, lr, decay=0.9):
    layers = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    param_groups = []
    for i, (name, param) in enumerate(reversed(layers)):
        layer_lr = lr * (decay ** (i // 12))
        param_groups.append({"params": [param], "lr": layer_lr})
    return param_groups

def train_fold_engine(name, hf_path, train_df, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        print(f"\n🚀 Training Model Stack: {name} | Fold {fold+1}/{n_folds}")
        tok = AutoTokenizer.from_pretrained(hf_path)
        model = AutoModelForSequenceClassification.from_pretrained(hf_path, num_labels=2, ignore_mismatched_sizes=True).float().to(DEVICE)
        
        trn_ld = DataLoader(PairDS(train_df.iloc[trn_idx], tok, 256), batch_size=8, shuffle=True)
        opt = torch.optim.AdamW(get_llrd_params(model, 8e-6), weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda")
        crit = Focal()
        
        model.train()
        for ep in range(3):
            for b in trn_ld:
                opt.zero_grad()
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = crit(model(input_ids=b["input_ids"].to(DEVICE), attention_mask=b["attention_mask"].to(DEVICE)).logits, b["labels"].to(DEVICE))
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        
        output_path = f"/content/kaggle/working/{name}_fold{fold}.pt"
        torch.save(model.state_dict(), output_path)
        print(f"✅ Checkpoint written: {output_path}")
        del model, tok; gc.collect(); torch.cuda.empty_cache()

# 6. Run the 15-Model Stack Strategy
train_fold_engine("banglabert_large", "csebuetnlp/banglabert_large", train_master)
train_fold_engine("mdeberta", "microsoft/mdeberta-v3-base", train_master)
train_fold_engine("xlm_roberta", "joeddav/xlm-roberta-large-xnli", train_master)
print("\n🎉 Entire cross-validation blueprint completed!")





```

```python
!zip -j /content/trained_5fold_encoders.zip /content/kaggle/working/*.pt


```

