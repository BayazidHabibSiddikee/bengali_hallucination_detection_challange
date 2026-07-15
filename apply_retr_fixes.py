import json

with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    
    if "def build_retr_signal()" in source and "Sim=linear_kernel" in source:
        # We need to replace the scoring logic in score(df)
        
        old_score_logic = """        for r_,ti in zip(df.iloc[idx].itertuples(),top):
            for j in ti:
                prem.append(str(r_.prompt_bn)+" "+chunks[j]); resp.append(str(r_.response_bn))
        pp=predict_proba(model,tok,pd.DataFrame({"premise":prem,"response":resp}),cfg.max_len,cfg.batch_size*2)
"""
        # Find the line that assigns out[idx]
        lines = source.split("\n")
        new_lines = []
        skip = False
        for line in lines:
            if "for r_,ti in zip(df.iloc[idx].itertuples(),top):" in line:
                skip = True
                new_lines.append("""        for r_,ti in zip(df.iloc[idx].itertuples(),top):
            for j in ti:
                prem.append(str(r_.prompt_bn)+" "+chunks[j]); resp.append(str(r_.response_bn))
        pp=predict_proba(model,tok,pd.DataFrame({"premise":prem,"response":resp}),cfg.max_len,cfg.batch_size*2)
        scores_2d = pp.reshape(len(idx), cfg.retr_topk)
        
        MIN_SIM = 0.05
        weights = np.array([1/(i+1) for i in range(cfg.retr_topk)])
        
        for ri, ti in enumerate(top):
            valid_mask = Sim[ri, ti] >= MIN_SIM
            if not valid_mask.any(): valid_mask[0] = True
            w = weights * valid_mask
            scores_2d[ri] = scores_2d[ri] * (w / w.sum())
            
        out[idx] = scores_2d.sum(1)
        return out""")
            elif skip:
                if "return out" in line or "max(1)" in line or "mean(1)" in line:
                    skip = False # we're done skipping the old return logic
            else:
                new_lines.append(line)
                
        cell["source"] = [l + '\n' for l in new_lines[:-1]] + [new_lines[-1]] if new_lines else []
        
with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Applied advanced retrieval filtering and weighting!")
