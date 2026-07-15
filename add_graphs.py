import json
import shutil

# 1. Copy bengali-hallu.ipynb to pipeline.ipynb
shutil.copy("bengali-hallu.ipynb", "pipeline.ipynb")

# 2. Add Interactive Graphs Cell
with open("pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

graph_code = """# ===== CELL 18 — INTERACTIVE ERROR ANALYSIS & VISUALIZATIONS =====
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from IPython.display import display

print("Generating Interactive Visualizations...")

try:
    # 1. Probability Distribution by Label (Requires 'wrong' DataFrame from Cell 17)
    if 'wrong' in globals():
        fig_prob = px.histogram(wrong, x="prob", color="label", nbins=40,
                                title="Probability Distribution of Errors",
                                labels={"prob": "Predicted Probability", "label": "True Label"},
                                color_discrete_sequence=["#EF553B", "#00CC96"])
        fig_prob.update_layout(bargap=0.1)
        fig_prob.show()

    # 2. Ensemble Weights Comparison
    if 'wc' in globals() and 'wn' in globals():
        weights_df = pd.DataFrame({
            'Feature': list(wc.keys()),
            'Has Context': list(wc.values()),
            'No Context': list(wn.values())
        })
        fig_weights = px.bar(weights_df, x='Feature', y=['Has Context', 'No Context'], barmode='group',
                             title="Optimized Ensemble Weights (Has Context vs No Context)",
                             color_discrete_sequence=["#636EFA", "#FFA15A"])
        fig_weights.show()

    # 3. Source Breakdown of Sample (Val Set)
    if 'sample' in globals() and 'src' in sample.columns:
        fig_src = px.pie(sample, names='src', title="Validation Set Distribution by Source", hole=0.4)
        fig_src.show()
        
    # 4. Hallucination vs Faithful Distribution
    if 'sample' in globals() and 'label' in sample.columns:
        lbl_map = {0: "Hallucinated (0)", 1: "Faithful (1)"}
        dist_df = sample['label'].map(lbl_map).value_counts().reset_index()
        dist_df.columns = ['Label', 'Count']
        fig_dist = px.bar(dist_df, x='Label', y='Count', title="Overall Validation Label Distribution", color='Label')
        fig_dist.show()
except Exception as e:
    print(f"Visualization error: {e}")
"""

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + '\n' for line in graph_code.split('\n')[:-1]] + [graph_code.split('\n')[-1]]
}

# Append before the final tleft() cell if it exists, or just at the end.
# Actually, just append it at the very end.
nb["cells"].append(new_cell)

with open("pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated pipeline.ipynb with bengali-hallu.ipynb contents + interactive graphs.")
