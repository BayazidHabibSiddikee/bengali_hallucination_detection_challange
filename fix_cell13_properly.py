import json

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    src = "".join(cell.get("source", []))
    if "CELL 13 — TIGERLLM-9B JUDGE" in src:
        lines = src.split('\n')
        new_lines = []
        
        # We need to extract is_math_or_logic and get_category and their bodies,
        # and unindent them by 4 spaces.
        extracted_funcs = []
        inside_target_func = False
        
        for line in lines:
            if line.startswith("    def is_math_or_logic") or line.startswith("    def get_category"):
                inside_target_func = True
                extracted_funcs.append(line[4:])
                continue
            
            if inside_target_func:
                if line.strip() == "" or line.startswith("        ") or line.startswith("    #"):
                    if len(line) >= 4 and not line.strip() == "":
                        extracted_funcs.append(line[4:])
                    elif line.strip() == "":
                        extracted_funcs.append("")
                else:
                    # We reached the end of the function (e.g. def build_sys_prompt)
                    inside_target_func = False
                    new_lines.append(line)
            else:
                new_lines.append(line)
                
        # Now insert the extracted functions right before def run_llm_judge(df):
        final_lines = []
        for line in new_lines:
            if line.startswith("def run_llm_judge(df):"):
                final_lines.extend(extracted_funcs)
                final_lines.append(line)
            else:
                final_lines.append(line)
                
        cell["source"] = [l + "\n" if i < len(final_lines)-1 else l for i, l in enumerate(final_lines)]
        break

with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("✅ Properly fixed scoping in Cell 13!")
