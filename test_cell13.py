import re, os
import numpy as np

def is_no_ctx(ctx):
    return False

def is_math_or_logic(prompt, ctx):
    math_terms = ["কত", "যোগ", "বিয়োগ", "গুণ", "ভাগ", "শতকরা", "শতাংশ", "গণিত", "হিসাব", "সংখ্যা"]
    if any(m in str(prompt) for m in math_terms): return True
    if re.search(r'\d+', str(prompt)): return True
    return False

# --- get_category now lives at MODULE level so both run_llm_judge() and
# --- _categorize_df() (used later for the LightGBM category feature) can see it.
def get_category(prompt_text, ctx_text, response_text):
    p = str(prompt_text)
    combined = f"{p} {ctx_text} {response_text}"

    # 1. Code-mixed / Banglish (contains meaningful Latin chars)
    if re.search(r'[a-zA-Z]{2,}', combined):
        return "code_mixed"

    # 2. Has a real context passage → comprehension task
    if ctx_text and not is_no_ctx(ctx_text) and len(str(ctx_text).strip()) > 10:
        return "comprehension"

    # 3. Bengali History (C2 domain - important for scoring)
    history_kws = ["ইতিহাস", "সাল", "যুদ্ধ", "মুক্তিযুদ্ধ", "বিশ্বযুদ্ধ",
                   "শতক", "রাজত্ব", "সম্রাট", "জন্মগ্রহণ", "মৃত্যুবরণ",
                   "প্রতিষ্ঠিত", "আবিষ্কার", "বিপ্লব", "স্বাধীনতা"]
    if any(k in p for k in history_kws):
        return "history"

    # 4. Bengali vocabulary / language (C1 cultural-distance domain)
    vocab_kws = ["অর্থ", "ভাবার্থ", "সমার্থক", "বিপরীত", "মানে", "বাগধারা",
                 "ব্যাকরণ", "প্রতিশব্দ", "বিপরীতার্থক", "সন্ধি", "উপসর্গ",
                 "প্রত্যয়", "সমাস", "কারক", "বচন"]
    if any(k in p for k in vocab_kws):
        return "vocabulary"

    # 5. Math / logic / quantitative
    if is_math_or_logic(prompt_text, ""):
        return "math"

    # 6. Default: general knowledge (no context, not vocabulary, not math)
    return "general_knowledge"

print("Syntax is valid!")
