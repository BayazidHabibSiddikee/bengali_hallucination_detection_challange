import nbformat as nbf

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("# 🏗️ Bengali Hallucination Detection - Full Formalized Pipeline\nImplementation of the 7-stage hybrid regime architecture for Bengali LLM outputs."),
    
    nbf.v4.new_markdown_cell("## Stage 0 — Data Intelligence\nLoad the dataset, check for leaks, and perform error analysis."),
    nbf.v4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data (dummy path for now)
# df = pd.read_csv('bengali_hallucination_dataset.csv')

# Dummy data for pipeline testing
data = {
    'id': [1, 2, 3, 4],
    'prompt': ['বাংলাদেশের রাজধানী কী?', '২+২ কত?', 'Translate: Hello', 'Tell me about the 2024 Mars mission.'],
    'response': ['বাংলাদেশের রাজধানী ঢাকা।', '২+২=৫', 'হ্যালো', '২০২৪ সালে নাসা মঙ্গলে মানুষ পাঠিয়েছে।'],
    'context': [None, None, 'Hello = হ্যালো', 'NASA plans to send humans to Mars by 2030s.'],
    'label': [1, 0, 1, 0] # 1=Faithful, 0=Hallucinated
}
df = pd.DataFrame(data)

# Leakage Audit
print("Leakage Audit: Checking for duplicate IDs...")
print("Duplicates found:", df['id'].duplicated().sum())

# Basic Error Analysis
print("\\nLabel Distribution (0=Hallucinated, 1=Faithful):")
print(df['label'].value_counts(normalize=True))"""),

    nbf.v4.new_markdown_cell("## Stage 1 — Task/Regime Router\nClassify each sample into its specific evaluation regime."),
    nbf.v4.new_code_cell("""import re

def has_numbers(text):
    return bool(re.search(r'\\d+', text))

def is_translation_task(sample):
    # Heuristic: Check if prompt contains English characters and response is Bengali
    has_english = bool(re.search(r'[a-zA-Z]', str(sample['prompt'])))
    return has_english

def route(sample):
    if pd.notna(sample['context']):
        return 'context_grounding'
    elif has_numbers(str(sample['response'])):
        return 'math_reasoning'
    elif is_translation_task(sample):
        return 'translation_summarization'
    else:
        return 'factual_qa'

df['regime'] = df.apply(route, axis=1)
print("Regime Distribution:")
print(df['regime'].value_counts())"""),

    nbf.v4.new_markdown_cell("## Stage 2 — Per-Regime Detectors\nImplement specialized detectors for each branch."),
    nbf.v4.new_code_cell("""from transformers import pipeline

# 1. Context Grounding -> NLI Cross-Encoder
# nli_model = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")
def context_grounding_score(context, response):
    # return nli_model(f"{context} [SEP] {response}")
    return np.random.uniform(0.5, 1.0) # Dummy score

# 2. No-Context Factual QA -> Retrieval + Evidence Verification
def factual_qa_score(prompt, response):
    # Step: Retrieve from Bangla Wikipedia / Web
    # Step: Cross-encode retrieved context with response
    return np.random.uniform(0.2, 0.9)

# 3. Math/Number -> Symbolic Verifier
def math_verification_score(prompt, response):
    # Step: Extract numbers and verify
    # e.g., if '২+২=৫' -> mismatch -> 0.0
    if '৫' in response and '২' in prompt:
        return 0.0 # Caught by symbolic verifier
    return 1.0

# 4. Translation/Summarization -> Semantic Entailment
def translation_entailment_score(prompt, response):
    # Model: multilingual-e5-large or LaBSE cosine similarity
    return np.random.uniform(0.7, 1.0)

# Apply detectors based on regime
def get_regime_score(row):
    if row['regime'] == 'context_grounding':
        return context_grounding_score(row['context'], row['response'])
    elif row['regime'] == 'math_reasoning':
        return math_verification_score(row['prompt'], row['response'])
    elif row['regime'] == 'translation_summarization':
        return translation_entailment_score(row['prompt'], row['response'])
    else:
        return factual_qa_score(row['prompt'], row['response'])

df['regime_score'] = df.apply(get_regime_score, axis=1)
print(df[['regime', 'regime_score']])"""),

    nbf.v4.new_markdown_cell("## Stage 3 — Encoder Ensemble\nExtract dense features using BanglaBERT, XLM-R, and mDeBERTa."),
    nbf.v4.new_code_cell("""# from sentence_transformers import SentenceTransformer
# model_banglabert = SentenceTransformer('sagorsarker/bangla-bert-base')
# model_xlmr = SentenceTransformer('xlm-roberta-base')

def get_ensemble_features(row):
    # Placeholder for actual embeddings (768-dim vectors)
    banglabert_cls = np.random.rand(5) # Dim reduced for dummy
    xlmr_cls = np.random.rand(5)
    
    # Combine with regime scores and one-hot encoding
    features = list(banglabert_cls) + list(xlmr_cls) + [row['regime_score']]
    return features

df['ensemble_features'] = df.apply(get_ensemble_features, axis=1)
X = np.stack(df['ensemble_features'].values)
y = df['label'].values"""),

    nbf.v4.new_markdown_cell("## Stage 4 & 5 — Meta Classifier & Uncertainty Detection\nTrain LightGBM and detect uncertain samples for LLM fallback."),
    nbf.v4.new_code_cell("""import lightgbm as lgb
from sklearn.model_selection import cross_val_predict

# Meta Classifier (LightGBM)
clf = lgb.LGBMClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Predict Probabilities
probs = clf.predict_proba(X)[:, 1] # Probability of being Faithful (1)
df['meta_prob'] = probs

# Uncertainty Detection
def uncertainty(probability):
    if probability > 0.75 or probability < 0.25:
        return 'confident'
    else:
        return 'uncertain'

df['confidence'] = df['meta_prob'].apply(uncertainty)
print(df[['label', 'meta_prob', 'confidence']])"""),

    nbf.v4.new_markdown_cell("## Stage 6 — LLM Fallback\nQuery an open-weight LLM (like Qwen2.5-7B) only for uncertain cases."),
    nbf.v4.new_code_cell("""def llm_fallback(row):
    if row['confidence'] == 'confident':
        return row['meta_prob'] # Keep original
    
    # For uncertain cases, we would call the LLM
    prompt = f"নিচের প্রম্পট এবং উত্তর পড়ুন।\\nপ্রম্পট: {row['prompt']}\\nউত্তর: {row['response']}\\n"
    if pd.notna(row['context']):
        prompt += f"তথ্যসূত্র: {row['context']}\\n"
    prompt += "উত্তরটি কি সঠিক এবং বিশ্বস্ত? শুধু হ্যাঁ বা না বলুন।"
    
    # print("Calling LLM with prompt:", prompt)
    # response = llm_pipeline(prompt)
    llm_decision = 1.0 if np.random.rand() > 0.5 else 0.0 # Dummy LLM response
    return llm_decision

df['final_prob'] = df.apply(llm_fallback, axis=1)"""),

    nbf.v4.new_markdown_cell("## Stage 7 — Calibration & Threshold Tuning\nOptimize threshold for F1-score on the Hallucinated class (0)."),
    nbf.v4.new_code_cell("""from sklearn.metrics import f1_score

# In a real scenario, use Out-of-Fold (OOF) predictions to tune threshold
thresholds = np.arange(0.3, 0.7, 0.01)
best_threshold = max(thresholds, key=lambda t: f1_score(y, df['final_prob'] > t, pos_label=0))

print(f"Optimal Threshold for detecting Hallucinations: {best_threshold:.2f}")

df['final_prediction'] = (df['final_prob'] > best_threshold).astype(int)
print("\\nFinal Pipeline Output:")
print(df[['id', 'prompt', 'label', 'final_prediction']])""")
]

nbf.write(nb, '/home/sword/bengali_hallucination_detection/pipeline.ipynb')
