import json
import re

def update_colab():
    with open('adding_multifold_training.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # Inject get_llrd_params
            if 'def train_fold_engine' in source and 'def get_llrd_params' not in source:
                llrd_code = '''def get_llrd_params(model, lr, decay=0.9):
    layers = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    param_groups = []
    for i, (name, param) in enumerate(reversed(layers)):
        layer_lr = lr * (decay ** (i // 12))
        param_groups.append({"params": [param], "lr": layer_lr})
    return param_groups

def train_fold_engine'''
                source = source.replace('def train_fold_engine', llrd_code)
                
            # Update AdamW
            if 'opt = torch.optim.AdamW(model.parameters(), lr=8e-6)' in source:
                source = source.replace(
                    'opt = torch.optim.AdamW(model.parameters(), lr=8e-6)',
                    'opt = torch.optim.AdamW(get_llrd_params(model, 8e-6), weight_decay=0.01)'
                )
                
            # Remove weak models
            if 'train_fold_engine("bangla_bert_base"' in source:
                # Remove lines containing bangla_bert_base or l3cube
                lines = source.split('\n')
                lines = [l for l in lines if 'train_fold_engine("bangla_bert_base"' not in l and 'train_fold_engine("l3cube"' not in l]
                source = '\n'.join(lines)
                
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source'] and cell['source'][-1].endswith('\n\n'):
                cell['source'][-1] = cell['source'][-1][:-1]
                
    with open('adding_multifold_training.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

def update_pipeline():
    with open('pipeline.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            if 'cfg.backbones =' in source:
                # Re-write the backbones block
                old_block_start = source.find('cfg.backbones = (')
                old_block_end = source.find(')', old_block_start) + 1
                
                new_block = '''cfg.backbones = (
    ("banglabert_large", "csebuetnlp/banglabert_large"),
    ("mdeberta", "microsoft/mdeberta-v3-base"),
    ("xlm_roberta", "joeddav/xlm-roberta-large-xnli")
)'''
                if old_block_start != -1:
                    source = source[:old_block_start] + new_block + source[old_block_end:]
                    
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source'] and cell['source'][-1].endswith('\n\n'):
                cell['source'][-1] = cell['source'][-1][:-1]
                
    with open('pipeline.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    update_colab()
    update_pipeline()
