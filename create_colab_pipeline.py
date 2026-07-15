import json
import shutil

def create_colab_pipeline():
    # Copy pipeline
    shutil.copyfile('pipeline.ipynb', 'pipeline_colab.ipynb')
    
    with open('pipeline_colab.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            if 'cfg.backbones =' in source:
                # Re-write the backbones block to include all 5
                old_block_start = source.find('cfg.backbones = (')
                old_block_end = source.find(')', old_block_start) + 1
                
                new_block = '''cfg.backbones = (
    ("banglabert_large", "csebuetnlp/banglabert_large"),
    ("mdeberta", "microsoft/mdeberta-v3-base"),
    ("bangla_bert_base", "sagorsarker/bangla-bert-base"),
    ("xlm_roberta", "joeddav/xlm-roberta-large-xnli"),
    ("l3cube", "l3cube-pune/bengali-bert")
)'''
                if old_block_start != -1:
                    source = source[:old_block_start] + new_block + source[old_block_end:]
                    
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source'] and cell['source'][-1].endswith('\n\n'):
                cell['source'][-1] = cell['source'][-1][:-1]
                
    with open('pipeline_colab.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

def convert(ipynb_path, md_path):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    with open(md_path, 'w', encoding='utf-8') as out:
        for cell in nb.get('cells', []):
            cell_type = cell.get('cell_type')
            source = ''.join(cell.get('source', []))
            if cell_type == 'markdown':
                out.write(source + '\n\n')
            elif cell_type == 'code':
                out.write('\`\`\`python\n' + source + '\n\`\`\`\n\n')

if __name__ == '__main__':
    create_colab_pipeline()
    convert('pipeline_colab.ipynb', 'pipeline_colab.md')
