import json

def update_colab():
    with open('adding_multifold_training.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # 1. Add PSEUDO_PATH
            if 'BHE_FULL    = find_file("/content/**/banglahallueval_qa_dataset.csv")' in source:
                old_str1 = 'BHE_FULL    = find_file("/content/**/banglahallueval_qa_dataset.csv")'
                new_str1 = 'BHE_FULL    = find_file("/content/**/banglahallueval_qa_dataset.csv")\nPSEUDO_PATH = find_file("/content/**/pseudo_labels.csv")'
                source = source.replace(old_str1, new_str1)
                
            # 2. Add merging logic
            if 'train_master = pd.concat([sample[["premise", "response", "label", "src"]], pd.DataFrame(bhe_rows, columns=["premise", "response", "label", "src"])]).dropna()' in source:
                old_str2 = 'train_master = pd.concat([sample[["premise", "response", "label", "src"]], pd.DataFrame(bhe_rows, columns=["premise", "response", "label", "src"])]).dropna()'
                new_str2 = '''pseudo_rows = []
if PSEUDO_PATH and os.path.exists(PSEUDO_PATH):
    df_pseudo = pd.read_csv(PSEUDO_PATH)
    if {"premise", "response", "label"}.issubset(df_pseudo.columns):
        df_pseudo["src"] = "pseudo"
        pseudo_rows.append(df_pseudo[["premise", "response", "label", "src"]])
        print(f"✓ Injected {len(df_pseudo)} Pseudo-Labels into training!")

train_master = pd.concat([
    sample[["premise", "response", "label", "src"]], 
    pd.DataFrame(bhe_rows, columns=["premise", "response", "label", "src"])
] + pseudo_rows).dropna()'''
                source = source.replace(old_str2, new_str2)
                
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source'] and cell['source'][-1].endswith('\n\n'):
                cell['source'][-1] = cell['source'][-1][:-1]
                
    with open('adding_multifold_training.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    update_colab()
