import json

def add_test_signals(filename):
    with open(filename, 'r') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = cell.get('source', [])
            if any('val_signals.csv' in line for line in src):
                new_src = []
                for line in src:
                    new_src.append(line)
                    if 'diag.to_csv("/kaggle/working/val_signals.csv",index=False)' in line:
                        new_src.append('diag_test=pd.DataFrame({"regime":np.where(test.no_ctx,"no","has"),"p":pt})\n')
                        new_src.append('for c in [c for c in Xt.columns if c!="no_ctx"]: diag_test[c]=Xt[c].values\n')
                        new_src.append('diag_test.to_csv("/kaggle/working/test_signals.csv",index=False)\n')
                        new_src.append('print("saved test_signals.csv")\n')
                
                nb['cells'][i]['source'] = new_src

    with open(filename, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Added test_signals.csv export to {filename}")

add_test_signals('pipeline.ipynb')
add_test_signals('bengali-hallu.ipynb')
