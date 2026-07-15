import json
import re

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    src = "".join(cell.get("source", []))
    if "CELL 13 — TIGERLLM-9B JUDGE" in src:
        # We need to un-indent the definitions of get_category and is_math_or_logic 
        # and move them to the top of the cell.
        
        # The easiest way is to just replace the structure.
        
        # Look for def is_math_or_logic and get_category and their code block
        # Instead of parsing perfectly, let's just do a string replacement to fix the scoping
        
        # Let's find the start of run_llm_judge
        # And just move it down
        
        # Actually, let's just make the whole cell content fixed manually
        new_source = src.replace(
            "def run_llm_judge(df):\n    dev = \"cuda:0\" if torch.cuda.is_available() else \"cpu\"\n    print(f\"free per GPU: {get_gpu_free()} -> preferring {dev}\")\n\n    def is_math_or_logic(prompt, response):",
            "def is_math_or_logic(prompt, response):"
        )
        
        new_source = new_source.replace(
            "    def build_sys_prompt(category):",
            "def run_llm_judge(df):\n    dev = \"cuda:0\" if torch.cuda.is_available() else \"cpu\"\n    print(f\"free per GPU: {get_gpu_free()} -> preferring {dev}\")\n\n    def build_sys_prompt(category):"
        )
        
        # Un-indent is_math_or_logic and get_category
        lines = new_source.split('\n')
        fixed_lines = []
        inside_global_funcs = False
        for line in lines:
            if line.startswith("def is_math_or_logic"):
                inside_global_funcs = True
            elif line.startswith("def run_llm_judge(df):"):
                inside_global_funcs = False
                
            if inside_global_funcs and line.startswith("    "):
                fixed_lines.append(line[4:])
            else:
                fixed_lines.append(line)
                
        cell["source"] = [l + "\n" for l in fixed_lines]
        # Clean up the last newline
        if cell["source"]:
            cell["source"][-1] = cell["source"][-1][:-1]
            
        break

with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("✅ Fixed scoping issue in Cell 13! get_category is now accessible globally.")
