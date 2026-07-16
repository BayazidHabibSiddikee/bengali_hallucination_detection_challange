import json

def fix():
    with open('adding_multifold_training.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # Remove the redundant first cell if it's the one I added
    if '!mkdir -p /content/kaggle/input/datasets/bayazidhs/bengali-hallucination-data' in nb['cells'][0]['source'][0]:
        nb['cells'].pop(0)
        
    # Now find the setup cell and add the new dataset download
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'os.environ[\'KAGGLE_USERNAME\']' in "".join(cell['source']):
            source = "".join(cell['source'])
            
            # Add the mkdir if it's missing
            if 'bengali-trained-mdeberta' not in source:
                old_mkdir = '!mkdir -p /content/kaggle/input/bangla-wikipedia-dataset\n'
                new_mkdir = '!mkdir -p /content/kaggle/input/bangla-wikipedia-dataset\n!mkdir -p /content/kaggle/input/datasets/bayazidhs/bengali-trained-mdeberta\n!mkdir -p /kaggle/working/\n'
                source = source.replace(old_mkdir, new_mkdir)
                
                # Add the kaggle download command for pseudo labels
                old_dl = '!kaggle datasets download mahdihasanqurishi/banglahallueval-qa -p /content/kaggle/input/datasets/mahdihasanqurishi/banglahallueval-qa --unzip\n'
                new_dl = old_dl + '\nprint("Downloading Pseudo Labels...")\n!kaggle datasets download bayazidhs/bengali-trained-mdeberta -p /content/kaggle/input/datasets/bayazidhs/bengali-trained-mdeberta --unzip\n'
                source = source.replace(old_dl, new_dl)
                
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source'] and cell['source'][-1].endswith('\n\n'):
                cell['source'][-1] = cell['source'][-1][:-1]
                
    with open('adding_multifold_training.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    fix()
