"""
Fix the get_category() function in Cell 13 to correctly detect:
- Bengali History (ইতিহাস, সাল, যুদ্ধ, মুক্তিযুদ্ধ) -> 'history'  
- Bengali culture/language (ভাবার্থ, বাগধারা, ব্যাকরণ) -> 'vocabulary'
- No-context knowledge (knowledge without any passage) -> 'general_knowledge'
- Math -> 'math'
- Comprehension (has context passage) -> 'comprehension'
- Banglish/Code-mixed -> 'code_mixed'
"""
import json

def fix_get_category(filename):
    with open(filename) as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src_str = ''.join(cell.get('source', []))
        if 'def get_category' in src_str:
            # Replace the get_category function with a more robust version
            OLD = '''    def get_category(prompt_text, ctx_text, response_text):
        p_lower = str(prompt_text).lower()
        combined_text = f"{prompt_text} {ctx_text} {response_text}"
        if re.search(r'[a-zA-Z]', combined_text):
            return "code_mixed"
        elif ctx_text and not is_no_ctx(ctx_text):
            return "comprehension"
        elif any(k in p_lower for k in ["অর্থ", "ভাবার্থ", "সমার্থক", "বিপরীত",
"মানে কী"]):
            return "vocabulary"
        elif is_math_or_logic(prompt_text, ""):
            return "math"
        else:
            return "general_knowledge"'''

            NEW = '''    def get_category(prompt_text, ctx_text, response_text):
        p = str(prompt_text)
        p_lower = p.lower()
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
        return "general_knowledge"'''

            # Try to replace, but handle newline differences
            src = cell.get('source', [])
            src_joined = ''.join(src)
            
            # Find and replace the function block
            if 'def get_category' in src_joined and 'return "general_knowledge"' in src_joined:
                # Replace line by line approach
                new_src = []
                in_get_category = False
                skip_until_return = False
                
                for line in src:
                    if '    def get_category(' in line:
                        in_get_category = True
                        # Replace entire function
                        for new_line in NEW.split('\n'):
                            new_src.append(new_line + '\n')
                        skip_until_return = True
                        continue
                    
                    if skip_until_return:
                        if 'return "general_knowledge"' in line:
                            skip_until_return = False
                            in_get_category = False
                        continue
                    
                    new_src.append(line)
                
                nb['cells'][i]['source'] = new_src
                print(f"Fixed get_category in {filename} cell {i}")

    with open(filename, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

fix_get_category('pipeline.ipynb')
fix_get_category('bengali-hallu.ipynb')
