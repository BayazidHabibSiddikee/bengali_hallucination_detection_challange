import json

def apply_ratio(filename):
    with open(filename, 'r') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = cell.get('source', [])
            if any('k=min(len(c1),len(c0))' in line for line in src):
                new_src = []
                skip = False
                for line in src:
                    if 'k=min(len(c1),len(c0))' in line:
                        skip = True
                        new_src.append('MAX_RATIO = 2.0\n')
                        new_src.append('k0 = min(len(c0), int(len(c1) * MAX_RATIO))\n')
                        new_src.append('k1 = min(len(c1), int(len(c0) * MAX_RATIO))\n')
                        new_src.append('if len(c0)>k0:\n')
                        new_src.append('    c0=(c0.groupby("mode",group_keys=False)\n')
                        new_src.append('          .apply(lambda g:g.sample(max(1,int(round(k0*len(g)/len(train_all[train_all.label==0])))),random_state=SEED)))\n')
                        new_src.append('    c0=c0.sample(min(len(c0),k0),random_state=SEED)\n')
                        new_src.append('if len(c1)>k1: c1=c1.sample(k1,random_state=SEED)\n')
                        continue
                    
                    if skip and 'train_all=pd.concat' in line:
                        skip = False
                        new_src.append('train_all=pd.concat([c1,c0]).sample(frac=1,random_state=SEED).reset_index(drop=True)\n')
                        continue
                    
                    if skip:
                        continue
                        
                    new_src.append(line)
                
                nb['cells'][i]['source'] = new_src

    with open(filename, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Applied 1:2 ratio to {filename}")

apply_ratio('pipeline.ipynb')
apply_ratio('bengali-hallu.ipynb')
