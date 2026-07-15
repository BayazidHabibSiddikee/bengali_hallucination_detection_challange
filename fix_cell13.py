import json

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        
        if "def run_llm_judge(df):" in source and "TigerLLM-9B-it" in source:
            if "def is_math_or_logic" not in source:
                # Add the function right below import re
                target = "import re"
                func_code = """
    def is_math_or_logic(prompt, ctx):
        math_terms = ["কত", "যোগ", "বিয়োগ", "গুণ", "ভাগ", "শতকরা", "শতাংশ", "গণিত", "হিসাব", "সংখ্যা"]
        if any(m in str(prompt) for m in math_terms): return True
        import re as local_re
        if local_re.search(r'\d+', str(prompt)): return True
        return False
"""
                new_source = source.replace(target, target + "\n" + func_code)
                cell["source"] = [new_source]
            
with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

