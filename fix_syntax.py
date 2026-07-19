import json
import ast

with open("pipeline.ipynb") as f:
    nb = json.load(f)

for i, c in enumerate(nb["cells"]):
    if c.get("cell_type") == "code":
        src = "".join(c.get("source", []))
        
        if "microsoft/mdeberta-v3-base" in src and "cfg.backbones" in src:
            # We fix the duplicate tuple
            lines = src.splitlines(keepends=True)
            new_lines = []
            skip = False
            for line in lines:
                if line.strip() == "," and "microsoft/mdeberta-v3-base" in lines[lines.index(line)+1]:
                    skip = True
                if skip and line.strip() == ")":
                    skip = False
                    continue
                if not skip:
                    new_lines.append(line)
            c["source"] = new_lines
            print("Fixed Cell 9.")
            
        src_after_9 = "".join(c.get("source", []))
        if "print(f\"\\n{=*60}\")" in src_after_9:
            new_src = src_after_9.replace("print(f\"\\n{=*60}\")", "print(f\"\\n{'='*60}\")")
            c["source"] = new_src.splitlines(keepends=True)
            print("Fixed Cell 13.")

with open("pipeline.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

full_code = ""
for i, c in enumerate(nb["cells"]):
    if c.get("cell_type") == "code":
        src = "".join(c.get("source", []))
        clean = "\n".join(l if not l.strip().startswith(("!","%")) else f"#{l}" for l in src.splitlines())
        full_code += clean + "\n"

try:
    ast.parse(full_code)
    print("Syntax verification SUCCESS!")
except SyntaxError as e:
    print(f"Syntax verification FAILED: {e}")
