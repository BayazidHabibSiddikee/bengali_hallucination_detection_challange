import json
import re

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find Cell 13 and rewrite it for Dual LLM
for cell in nb.get("cells", []):
    src = "".join(cell.get("source", []))
    if "CELL 13 — TIGERLLM-9B JUDGE" in src:
        new_source = """# ===== CELL 13 — DUAL-LLM JUDGE (TIGERLLM-9B + QWEN2.5-3B) =====
cfg.llm_input_len = 512

import re, os
import gc, torch
import numpy as np

def is_math_or_logic(prompt, ctx):
    math_terms = ["কত", "যোগ", "বিয়োগ", "গুণ", "ভাগ", "শতকরা", "শতাংশ", "গণিত", "হিসাব", "সংখ্যা"]
    if any(m in str(prompt) for m in math_terms): return True
    if re.search(r'\d+', str(prompt)): return True
    return False

def get_category(prompt_text, ctx_text, response_text):
    p = str(prompt_text)
    combined = f"{p} {ctx_text} {response_text}"
    if re.search(r'[a-zA-Z]{2,}', combined): return "code_mixed"
    if ctx_text and not is_no_ctx(ctx_text) and len(str(ctx_text).strip()) > 10: return "comprehension"
    history_kws = ["ইতিহাস", "সাল", "যুদ্ধ", "মুক্তিযুদ্ধ", "বিশ্বযুদ্ধ", "শতক", "রাজত্ব", "সম্রাট", "জন্মগ্রহণ", "মৃত্যুবরণ", "প্রতিষ্ঠিত", "আবিষ্কার", "বিপ্লব", "স্বাধীনতা"]
    if any(k in p for k in history_kws): return "history"
    vocab_kws = ["অর্থ", "ভাবার্থ", "সমার্থক", "বিপরীত", "মানে", "বাগধারা", "ব্যাকরণ", "প্রতিশব্দ", "বিপরীতার্থক", "সন্ধি", "উপসর্গ", "প্রত্যয়", "সমাস", "কারক", "বচন"]
    if any(k in p for k in vocab_kws): return "vocabulary"
    if is_math_or_logic(prompt_text, ""): return "math"
    return "general_knowledge"

def build_sys_prompt(category):
    examples = {
        "comprehension": ("প্রশ্ন: মেহদী হাসান খান কোন বিশ্ববিদ্যালয়ের ছাত্র?\\nঅনুচ্ছেদ: মেহদী হাসান খান ময়মনসিংহ মেডিকেল কলেজের একজন ছাত্র।\\nউত্তর: ময়মনসিংহ মেডিকেল কলেজ → Verdict: 1\\nউত্তর: ঢাকা বিশ্ববিদ্যালয় → Verdict: 0"),
        "vocabulary": ("প্রশ্ন: 'কাঁচা সোনা' বাগধারার অর্থ কী?\\nউত্তর: অপরিশোধিত স্বর্ণ → Verdict: 1\\nউত্তর: তাজা শাকসবজি → Verdict: 0"),
        "history": ("প্রশ্ন: বাংলাদেশের মুক্তিযুদ্ধ কত সালে হয়?\\nউত্তর: ১৯৭১ সালে → Verdict: 1\\nউত্তর: ১৯৪৭ সালে → Verdict: 0"),
        "math": ("প্রশ্ন: ২৫ এর ২০% কত?\\nউত্তর: ৫ → Verdict: 1\\nউত্তর: ৫০ → Verdict: 0"),
        "code_mixed": ("প্রশ্ন: Python-এ list এর length বের করার function কী?\\nউত্তর: len() → Verdict: 1\\nউত্তর: size() → Verdict: 0"),
        "general_knowledge": ("প্রশ্ন: বাংলাদেশের রাজধানী কোথায়?\\nউত্তর: ঢাকা → Verdict: 1\\nউত্তর: চট্টগ্রাম → Verdict: 0"),
    }
    base_rule = ("কোনো ব্যাখ্যা দেবেন না। শুধুমাত্র 0 অথবা 1 লিখুন।\\nউদাহরণ:\\n" + examples.get(category, examples["general_knowledge"]))
    if category == "code_mixed": return "আপনি একজন বহুভাষিক হ্যালুসিনেশন বিশ্লেষক। প্রশ্ন বা উত্তরে বাংলা, ইংরেজি বা বাংলিশের মিশ্রণ থাকতে পারে। উত্তরটি সঠিক হলে শুধু '1', ভুল হলে শুধু '0' লিখুন। " + base_rule
    elif category == "comprehension": return "আপনি একটি নির্ভুল হ্যালুসিনেশন সনাক্তকরণ এআই। দেওয়া অনুচ্ছেদ থেকে প্রশ্নের উত্তরটি সঠিকভাবে যাচাই করুন। অনুচ্ছেদের তথ্যের সাথে মিলে গেলে '1', না মিললে বা অতিরিক্ত তথ্য থাকলে '0' লিখুন। " + base_rule
    elif category == "vocabulary": return "আপনি একজন বিশেষজ্ঞ বাংলা ভাষাবিদ। বাংলা শব্দ, বাগধারা, ব্যাকরণ বা ভাষাতাত্ত্বিক প্রশ্নের উত্তর যাচাই করুন। এটি C1 সাংস্কৃতিক-ভাষাগত বিভাগ। সঠিক হলে '1', ভুল হলে '0' লিখুন। " + base_rule
    elif category == "history": return "আপনি একজন বাংলাদেশ ও বাংলা ইতিহাসের বিশেষজ্ঞ। ঐতিহাসিক তথ্য, সাল, ঘটনা ও ব্যক্তিত্ব যাচাই করুন। তথ্য সঠিক হলে '1', ভুল বা বানোয়াট হলে '0' লিখুন। " + base_rule
    elif category == "math": return "আপনি একজন গাণিতিক মূল্যায়নকারী এআই। গাণিতিক হিসাব ও উত্তর নিখুঁতভাবে যাচাই করুন। সম্পূর্ণ সঠিক হলে '1', সামান্যতম ভুল থাকলে '0' লিখুন। " + base_rule
    else: return "আপনি একজন কঠোর তথ্য-যাচাইকারী। সাধারণ জ্ঞান ও বাস্তবিক তথ্যের সত্যতা যাচাই করুন। সম্পূর্ণ সত্য ও নির্ভুল হলে '1', ভুল বা মনগড়া হলে '0' লিখুন। " + base_rule

def run_single_llm(df, model_path, tk_path, target_gpu, is_qwen=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    # Clean memory before loading
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    
    print(f"\\n--- Loading LLM from {model_path} on GPU {target_gpu} ---")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tk = AutoTokenizer.from_pretrained(tk_path)
    
    try:
        # Pre-quantized TigerLLM check
        if os.path.exists(os.path.join(model_path, "config.json")) and "tiger" in model_path.lower():
            llm = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": target_gpu}).eval()
        else:
            llm = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb, device_map={"": target_gpu}).eval()
        print(f"✅ Loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load LLM: {e}")
        return np.full(len(df), np.nan)
        
    dev = next(llm.parameters()).device
    
    def digit_ids(d):
        ids = set()
        for s in (d, " " + d):
            e = tk.encode(s, add_special_tokens=False)
            if e: ids.add(e[-1])
        return list(ids)

    ids1, ids0 = digit_ids("1"), digit_ids("0")
    out = np.zeros(len(df))
    
    for i, r in enumerate(df.itertuples()):
        ctx = getattr(r, "ctx_clean", "")
        cat = get_category(r.prompt_bn, ctx, r.response_bn)
        SYS = build_sys_prompt(cat)
        u = (f"CONTEXT: {ctx}\\n" if ctx else "") + f"QUESTION: {r.prompt_bn}\\nANSWER: {r.response_bn}\\nVerdict:"
        
        enc = tk.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": u}],
                                     add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ii = enc["input_ids"][:, -cfg.llm_input_len:].to(dev)
        am = enc["attention_mask"][:, -cfg.llm_input_len:].to(dev)
        
        with torch.no_grad():
            lg = llm(input_ids=ii, attention_mask=am).logits[0, -1, :].float()
        
        p1 = torch.logsumexp(lg[ids1], 0)
        p0 = torch.logsumexp(lg[ids0], 0)
        out[i] = torch.softmax(torch.stack([p0, p1]), 0)[1].item()
        
        if i % 250 == 0:
            print(f"  Processed {i}/{len(df)} | Cat: {cat}")
            
    # NUKE the LLM from memory
    del llm, tk
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    return out

def run_llm_judge(df):
    if not cfg.use_llm_judge: return np.full(len(df), np.nan)
    
    free_by_gpu = {i: torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())}
    target_gpu = max(free_by_gpu, key=free_by_gpu.get)
    
    TIGER_PATH = "/kaggle/input/datasets/bayazidhs/tigerllm-9b-4bit/tigerllm-9b-4bit"
    
    # 1. Run TigerLLM
    print("🧠 BRAIN 1: Running TigerLLM-9B")
    out_tiger = run_single_llm(df, TIGER_PATH, TIGER_PATH, target_gpu)
    
    # 2. Run Qwen2.5-3B
    print("🧠 BRAIN 2: Running Qwen2.5-3B-Instruct")
    # Resolve Qwen path using the helper function (assuming resolve_model is defined in cell 2)
    QWEN_PATH = resolve_model("Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct") 
    out_qwen = run_single_llm(df, QWEN_PATH, QWEN_PATH, target_gpu, is_qwen=True)
    
    # 3. Average the brains
    if np.isnan(out_tiger).all() and np.isnan(out_qwen).all():
        return out_tiger
    elif np.isnan(out_qwen).all():
        return out_tiger
    elif np.isnan(out_tiger).all():
        return out_qwen
    else:
        print("✅ Dual-LLM Averaging Complete!")
        return (out_tiger + out_qwen) / 2.0

# ── Extract category labels for LightGBM meta-feature ──────────────────────
CATEGORY_MAP = {"comprehension": 0, "math": 1, "vocabulary": 2, "general_knowledge": 3, "history": 4, "code_mixed": 5}

def _categorize_df(df):
    cats = []
    for r in df.itertuples():
        ctx = getattr(r, "ctx_clean", "")
        cats.append(get_category(r.prompt_bn, ctx, r.response_bn))
    return cats

sample["category"] = _categorize_df(sample)
test["category"] = _categorize_df(test)
print("Val category dist:", sample["category"].value_counts().to_dict())

llm_val = run_llm_judge(sample)
llm_test = run_llm_judge(test)
tleft()
"""
        # Overwrite the cell
        cell["source"] = [line + "\n" for line in new_source.split('\n')]

with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("✅ Created Dual-LLM version of pipeline.ipynb!")
