import json
import re

with open("pipeline.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "def _find_comp_dir():" in src:
            # Replace the entire cell
            new_src = """# ============================================================
# Setup: environment detection, paths, configuration
# ============================================================
import importlib, subprocess, sys
for mod, pkg in [("pypdf", "pypdf"), ("lightgbm", "lightgbm"),
                 ("sentence_transformers", "sentence-transformers"),
                 ("accelerate", "accelerate")]:
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import os, re, gc, json, glob, time, warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
T0 = time.time()

def stamp(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

ON_KAGGLE = os.path.exists("/kaggle/input")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = DEVICE == "cuda"

def _find_comp_dir():
    cands = [
        "/kaggle/input/datasets/bayazidhs/bengali-hallucination-data",
        "/kaggle/input/bengali-hallucination-data",
        "/kaggle/input/bengali-hallucination",
        "/kaggle/input/competitions/bengali-hallucination",
        "./dev_data"
    ]
    for c in cands:
        if os.path.isdir(c):
            return c
    return "./dev_data"

COMP_DIR = _find_comp_dir()

def _find_file(substr, ext):
    if COMP_DIR is None:
        return None
    hits = [f for f in sorted(glob.glob(os.path.join(COMP_DIR, "*" + ext)))
            if substr in os.path.basename(f).lower()]
    return hits[0] if hits else None

TRAIN_JSON = _find_file("sample", ".json") or _find_file("", ".json")
TEST_CSV = _find_file("test", ".csv")
SAMPLE_SUB = _find_file("submission", ".csv")

def _find_books_dir():
    cands = [
        "/kaggle/input/datasets/bayazidhs/bengali-historical-books",
        "/kaggle/input/bengali-historical-books",
        "./data"
    ]
    for c in cands:
        if os.path.isdir(c) and glob.glob(os.path.join(c, "*.pdf")):
            return c
    return None

BOOKS_DIR = _find_books_dir()

# Full-size models on Kaggle GPU; small stand-ins for local CPU smoke tests.
NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
EMB_MODEL_NAME = ("sentence-transformers/LaBSE" if ON_KAGGLE
                  else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_CANDIDATES = (["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
                   "Qwen/Qwen2.5-1.5B-Instruct"]
                  if ON_KAGGLE else ["Qwen/Qwen2.5-0.5B-Instruct"])

print(f"ON_KAGGLE={ON_KAGGLE}  DEVICE={DEVICE}")
print(f"COMP_DIR   = {COMP_DIR}")
print(f"TRAIN_JSON = {TRAIN_JSON}")
print(f"TEST_CSV   = {TEST_CSV}")
print(f"SAMPLE_SUB = {SAMPLE_SUB}")
print(f"BOOKS_DIR  = {BOOKS_DIR}")
"""
            cell["source"] = [line + "\n" for line in new_src.split("\n")]
            # Remove the last newline from the last line to match standard format
            cell["source"][-1] = cell["source"][-1].rstrip("\n")

with open("pipeline.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
