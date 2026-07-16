import json

def fix_bug(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            if 'pseudo_df = pseudo_df.nlargest(' in source and 'key=lambda x:' in source:
                # Replace the broken nlargest call
                old_block = '''    pseudo_df = pseudo_df.nlargest(
        min(cfg.pseudo_label_n, len(pseudo_df)),
        key=lambda x: abs(pt[conf_mask] - tt[conf_mask])
    )'''
                new_block = '''    pseudo_df["_conf_score"] = abs(pt[conf_mask] - tt[conf_mask])
    pseudo_df = pseudo_df.nlargest(min(cfg.pseudo_label_n, len(pseudo_df)), "_conf_score")'''
                source = source.replace(old_block, new_block)
                
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source'] and cell['source'][-1].endswith('\n\n'):
                cell['source'][-1] = cell['source'][-1][:-1]
                
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    fix_bug('pipeline.ipynb')
    fix_bug('pipeline_colab.ipynb')
