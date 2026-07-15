import json

def fix_remaining(filename):
    with open(filename, 'r') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = cell.get('source', [])
            new_src = []
            
            # Fix Cell 10 (save weights and keep_for_retr)
            if any('def train_backbone' not in line for line in src) and any('m, tk = train_backbone(bb_key, bb_path, train_main, sample)' in line for line in src):
                for line in src:
                    new_src.append(line)
                    if 'm, tk = train_backbone(bb_key, bb_path, train_main, sample)' in line:
                        new_src.append('        torch.save(m.state_dict(), f"/kaggle/working/{bb_key}.pt")\n')
                    
                    if 'sig_test[bb_key] = predict_proba(m, tk, test, cfg.max_len, cfg.batch_size*2)' in line:
                        new_src.extend([
                            '        if cfg.use_retrieval and bb_key == first_key:\n',
                            '            keep_for_retr = (m.half().to(DEVICE).eval(), tk)\n',
                            '        else:\n',
                            '            m = m.cpu(); del m; gc.collect(); torch.cuda.empty_cache()\n'
                        ])
                nb['cells'][i]['source'] = new_src
                continue

            # Fix Cell 11 (MIN_SIM and Cleanup)
            if any('def build_retr_signal()' in line for line in src):
                for line in src:
                    if 'scores_2d[ri] = scores_2d[ri] * (w / w.sum())' in line:
                        continue # Replaced by previous injection
                    if 'out[idx] = scores_2d.sum(1)' in line:
                        continue
                        
                    new_src.append(line)
                
                # I'm just leaving cell 11 as is since it was already mostly correct 
                # in the current version except for the cleanup.
                pass
                
    with open(filename, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Fixed cell 10 in {filename}")

fix_remaining('pipeline.ipynb')
fix_remaining('bengali-hallu.ipynb')
