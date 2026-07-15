import json

def update_pipeline():
    with open('pipeline.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if '("xlm_roberta", "joeddav/xlm-roberta-large-xnli")' in source and 'cfg.backbones =' in source:
                old_bb = '("xlm_roberta", "joeddav/xlm-roberta-large-xnli")\n)'
                new_bb = '("xlm_roberta", "joeddav/xlm-roberta-large-xnli"),\n    ("l3cube", "l3cube-pune/bengali-bert")\n)'
                source = source.replace(old_bb, new_bb)
                
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source'] and cell['source'][-1].endswith('\n\n'):
                cell['source'][-1] = cell['source'][-1][:-1]
                
    with open('pipeline.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

def update_colab():
    with open('adding_multifold_training.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'train_fold_engine("xlm_roberta", "joeddav/xlm-roberta-large-xnli", train_master)' in source:
                old_str = 'train_fold_engine("xlm_roberta", "joeddav/xlm-roberta-large-xnli", train_master)'
                new_str = 'train_fold_engine("xlm_roberta", "joeddav/xlm-roberta-large-xnli", train_master)\ntrain_fold_engine("l3cube", "l3cube-pune/bengali-bert", train_master)'
                source = source.replace(old_str, new_str)
                
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source'] and cell['source'][-1].endswith('\n\n'):
                cell['source'][-1] = cell['source'][-1][:-1]
                
    with open('adding_multifold_training.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    update_pipeline()
    update_colab()
