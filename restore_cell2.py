import json

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Recreate the missing Cell 2
cell_2 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ===== CELL 2 — GLOBALS =====\n",
        "import os, gc, sys, re, unicodedata, warnings, torch, traceback\n",
        "import numpy as np, pandas as pd\n",
        "from datasets import load_dataset, Dataset\n",
        "from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding\n",
        "from transformers import AutoModelForCausalLM, BitsAndBytesConfig\n",
        "from sklearn.metrics import f1_score\n",
        "import lightgbm as lgb\n",
        "from IPython.display import HTML, display\n",
        "\n",
        "warnings.simplefilter('ignore')\n",
        "os.environ[\"TOKENIZERS_PARALLELISM\"] = \"false\"\n",
        "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
        "SEED = 42\n",
        "torch.manual_seed(SEED); np.random.seed(SEED)\n",
        "print(f\"device: {DEVICE} | gpus: {torch.cuda.device_count()}\")\n"
    ]
}

# Insert it immediately after Cell 1
nb["cells"].insert(1, cell_2)

with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("✅ Restored Cell 2 with all imports!")
