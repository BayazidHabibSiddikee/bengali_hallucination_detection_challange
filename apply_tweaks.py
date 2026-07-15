import json

def tweak_notebook(filename):
    with open(filename, 'r') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = cell.get('source', [])
            new_src = []
            
            for line in src:
                # 1. Update LightGBM Hyperparameters
                if 'learning_rate=0.01' in line and 'num_leaves=3' in line:
                    line = line.replace('learning_rate=0.01', 'learning_rate=0.02')
                    line = line.replace('num_leaves=3', 'num_leaves=5')
                    line = line.replace('min_data_in_leaf=15', 'min_data_in_leaf=10')
                
                # 2. Update THR_SHIFT
                if 'THR_SHIFT = 0.0' in line:
                    line = line.replace('THR_SHIFT = 0.0', 'THR_SHIFT = 0.05')
                
                # 3. Update few-shot prompt
                if 'base_rule = "কোনো ব্যাখ্যা দেবেন না। শুধুমাত্র একটি সংখ্যা আউটপুট দিন।"' in line:
                    line = '        base_rule = "কোনো ব্যাখ্যা দেবেন না। শুধুমাত্র একটি সংখ্যা আউটপুট দিন।\\n\\nউদাহরণ (Example):\\nপ্রশ্ন: বাংলাদেশের রাজধানী কি?\\nউত্তর: ঢাকা\\nVerdict: 1\\n\\nপ্রশ্ন: বাংলাদেশের রাজধানী কি?\\nউত্তর: দিল্লি\\nVerdict: 0"\n'
                
                new_src.append(line)
            
            nb['cells'][i]['source'] = new_src

    with open(filename, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Tweaked {filename}")

tweak_notebook('pipeline.ipynb')
tweak_notebook('bengali-hallu.ipynb')
