import json

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if 'class CFG:' in source and 'llm_id:str' in source:
            if 'llm_input_len:int' not in source:
                source = source.replace('llm_id:str="md-nishat-008/TigerLLM-9B-it"\n', 'llm_id:str="md-nishat-008/TigerLLM-9B-it"\n    llm_input_len:int=512\n')
                
                lines = source.split('\n')
                cell["source"] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Added llm_input_len to CFG")
