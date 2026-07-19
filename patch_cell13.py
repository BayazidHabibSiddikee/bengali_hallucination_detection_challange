import json

with open("pipeline.ipynb") as f:
    nb = json.load(f)

install_block = """# ===== CELL 13 — INSTALL DEPENDENCIES IF MISSING =====
import subprocess, sys

def ensure_pkg(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", pkg], check=True)
        print(f"✅ {pkg} installed")

ensure_pkg("bitsandbytes")
ensure_pkg("accelerate")

import importlib
for mod in ["bitsandbytes", "accelerate"]:
    try:
        importlib.import_module(mod)
    except:
        pass

"""

new_try_block = """    if use_prequant:
        try:
            # First try: load without re-quantizing (fastest)
            llm = AutoModelForCausalLM.from_pretrained(
                path_or_id,
                device_map="balanced",
                max_memory=max_mem,
            ).eval()
            print(f"  ✅ Loaded pre-quantized {model_title}")
        except ImportError:
            # bitsandbytes not available — install and retry
            print("  ⚠ bitsandbytes missing — installing...")
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install",
                           "-q", "-U", "bitsandbytes>=0.46.1"], check=True)
            import bitsandbytes  # noqa
            llm = AutoModelForCausalLM.from_pretrained(
                path_or_id,
                device_map="balanced",
                max_memory=max_mem,
            ).eval()
            print(f"  ✅ Loaded after installing bitsandbytes")
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
            print(f"  ✅ Loaded with CPU offload")"""

replaced = False
for c in nb["cells"]:
    if c.get("cell_type") == "code":
        src = "".join(c.get("source", []))
        if "DUAL-LLM JUDGE" in src and "def run_llm_subengine" in src:
            # Insert install block at top
            # Usually it starts with `# ===== CELL 13`
            lines = src.splitlines(keepends=True)
            if lines[0].startswith("# ===== CELL 13"):
                lines.insert(1, install_block)
            else:
                lines.insert(0, install_block)
            src = "".join(lines)
            
            # Now replace the use_prequant block
            # I will just find the block from "if use_prequant:" to the "else:" part
            import re
            pattern = re.compile(r"    if use_prequant:.*?    else:", re.DOTALL)
            src = pattern.sub(new_try_block + "\n    else:", src)
            
            c["source"] = src.splitlines(keepends=True)
            replaced = True
            break

if replaced:
    with open("pipeline.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print("SUCCESS")
else:
    print("FAILED")
