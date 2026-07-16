import json

def insert_mkdir_cell():
    with open('adding_multifold_training.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    mkdir_source = [
        "!mkdir -p /content/kaggle/input/datasets/bayazidhs/bengali-hallucination-data\n",
        "!mkdir -p /content/kaggle/input/datasets/ajmainmahtab/bangla-natural-language-inference-dataset\n",
        "!mkdir -p /content/kaggle/input/datasets/disisbig/bengali-wikipedia-articles\n",
        "!mkdir -p /content/kaggle/input/datasets/mahdihasanqurishi/banglahallueval-qa\n",
        "!mkdir -p /content/kaggle/input/bangla-wikipedia-dataset\n",
        "!mkdir -p /content/kaggle/input/datasets/bayazidhs/bengali-trained-mdeberta\n",
        "!mkdir -p /kaggle/working/\n"
    ]
    
    mkdir_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": mkdir_source
    }
    
    # Check if we already inserted it
    if nb['cells'] and nb['cells'][0]['cell_type'] == 'code' and '!mkdir -p' in nb['cells'][0]['source'][0]:
        nb['cells'][0] = mkdir_cell
    else:
        nb['cells'].insert(0, mkdir_cell)
        
    with open('adding_multifold_training.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    insert_mkdir_cell()
