import json
import re

def upgrade_pipeline():
    with open('pipeline.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # 1. Focal Loss Tuning
            if "focal_gamma:float=1.0; focal_alpha:float=1.0" in source:
                source = source.replace("focal_gamma:float=1.0; focal_alpha:float=1.0", "focal_gamma:float=2.0; focal_alpha:float=0.75")
                
            # 2. Add 4th Backbone
            if '("bangla_bert_base", "sagorsarker/bangla-bert-base")' in source and 'cfg.backbones =' in source:
                old_bb = '("bangla_bert_base", "sagorsarker/bangla-bert-base")\n)'
                new_bb = '("bangla_bert_base", "sagorsarker/bangla-bert-base"),\n    ("xlm_roberta", "joeddav/xlm-roberta-large-xnli")\n)'
                source = source.replace(old_bb, new_bb)
                
            # 3 & 4. LightGBM Updates
            if 'fit_lgbm(X, y, mask, seed=SEED):' in source:
                old_params = """    params = dict(
        objective="binary", metric="binary_logloss", verbosity=-1, seed=seed,
        learning_rate=0.02, num_leaves=5, min_data_in_leaf=10,
        feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
    )"""
                new_params = """    params = dict(
        objective="binary", metric="binary_logloss", verbosity=-1, seed=seed,
        learning_rate=0.02, num_leaves=7, min_data_in_leaf=15,
        feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
        lambda_l1=0.1, lambda_l2=0.5
    )"""
                source = source.replace(old_params, new_params)
                
                # Add categorical feature
                old_train_cv = "m_cv = lgb.train(params, lgb.Dataset(X_trn, label=y_trn), num_boost_round=35)"
                new_train_cv = "m_cv = lgb.train(params, lgb.Dataset(X_trn, label=y_trn, categorical_feature=['category_enc']), num_boost_round=35)"
                source = source.replace(old_train_cv, new_train_cv)
                
                old_train_full = "model = lgb.train(params, lgb.Dataset(Xr, label=yr), num_boost_round=35)"
                new_train_full = "model = lgb.train(params, lgb.Dataset(Xr, label=yr, categorical_feature=['category_enc']), num_boost_round=35)"
                source = source.replace(old_train_full, new_train_full)
                
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source'] and cell['source'][-1].endswith('\n\n'):
                cell['source'][-1] = cell['source'][-1][:-1]
                
    with open('pipeline.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
def upgrade_colab():
    with open('adding_multifold_training.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # Update backbone list
            if 'train_fold_engine("bangla_bert_base", "sagorsarker/bangla-bert-base", train_master)' in source:
                old_str = 'train_fold_engine("bangla_bert_base", "sagorsarker/bangla-bert-base", train_master)'
                new_str = 'train_fold_engine("bangla_bert_base", "sagorsarker/bangla-bert-base", train_master)\ntrain_fold_engine("xlm_roberta", "joeddav/xlm-roberta-large-xnli", train_master)'
                source = source.replace(old_str, new_str)
                
            # Inject Focal Loss
            if 'class PairDS(Dataset):' in source and 'class Focal(' not in source:
                old_ds = 'class PairDS(Dataset):'
                new_ds = '''class Focal(nn.Module):
    def __init__(self,gamma=2.0,alpha=0.75):
        super().__init__(); self.g=gamma; self.register_buffer("w",torch.tensor([alpha,1.0]))
    def forward(self,lg,y):
        ce=F.cross_entropy(lg,y,weight=self.w.to(lg.device),reduction="none")
        pt=torch.exp(-ce); return ((1-pt)**self.g*ce).mean()

class PairDS(Dataset):'''
                source = source.replace(old_ds, new_ds)
                
            # Update CrossEntropy to Focal
            if 'loss = F.cross_entropy(model(input_ids' in source:
                # Add crit before the loop
                if 'scaler = torch.amp.GradScaler("cuda")' in source:
                    source = source.replace('scaler = torch.amp.GradScaler("cuda")', 'scaler = torch.amp.GradScaler("cuda")\n        crit = Focal()')
                
                old_loss = 'loss = F.cross_entropy(model(input_ids=b["input_ids"].to(DEVICE), attention_mask=b["attention_mask"].to(DEVICE)).logits, b["labels"].to(DEVICE))'
                new_loss = 'loss = crit(model(input_ids=b["input_ids"].to(DEVICE), attention_mask=b["attention_mask"].to(DEVICE)).logits, b["labels"].to(DEVICE))'
                source = source.replace(old_loss, new_loss)
                
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source'] and cell['source'][-1].endswith('\n\n'):
                cell['source'][-1] = cell['source'][-1][:-1]
                
    with open('adding_multifold_training.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    upgrade_pipeline()
    upgrade_colab()
