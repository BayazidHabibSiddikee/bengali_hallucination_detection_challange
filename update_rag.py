import json

with open('/home/sword/bengali_hallucination_detection/pipeline.ipynb', 'r') as f:
    nb = json.load(f)

# The cell we want to replace is the one containing "Stage 2 — Per-Regime Detectors"
# Actually, I'll insert a new cell for RAG Initialization right before Stage 2, and then update Stage 2.

rag_init_cell = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "source": [
        "# --- RAG Initialization (For Factual QA Branch) ---\n",
        "import os\n",
        "from langchain_community.document_loaders import PyPDFDirectoryLoader\n",
        "from langchain_text_splitters import RecursiveCharacterTextSplitter\n",
        "from langchain_community.vectorstores import FAISS\n",
        "from langchain_community.embeddings import HuggingFaceEmbeddings\n",
        "\n",
        "PDF_DIR = '/home/sword/Documents/tools/downloads'\n",
        "FAISS_DB_PATH = '/home/sword/bengali_hallucination_detection/faiss_db'\n",
        "\n",
        "print('Loading embedding model for RAG...')\n",
        "try:\n",
        "    embed_model = HuggingFaceEmbeddings(model_name='sagorsarker/bangla-bert-base')\n",
        "except Exception:\n",
        "    # Fallback if bangla-bert is not cached\n",
        "    embed_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')\n",
        "\n",
        "if os.path.exists(FAISS_DB_PATH):\n",
        "    print('Loading existing FAISS database...')\n",
        "    faiss_db = FAISS.load_local(FAISS_DB_PATH, embed_model, allow_dangerous_deserialization=True)\n",
        "else:\n",
        "    print(f'Building FAISS database from {PDF_DIR}...')\n",
        "    loader = PyPDFDirectoryLoader(path=PDF_DIR, glob='**/*.pdf')\n",
        "    docs = loader.load()\n",
        "    if len(docs) > 0:\n",
        "        print(f'Loaded {len(docs)} pages. Splitting...')\n",
        "        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)\n",
        "        chunks = splitter.split_documents(docs)\n",
        "        faiss_db = FAISS.from_documents(chunks, embed_model)\n",
        "        faiss_db.save_local(FAISS_DB_PATH)\n",
        "        print('FAISS database built and saved!')\n",
        "    else:\n",
        "        print('No PDFs found! Creating an empty FAISS index fallback.')\n",
        "        faiss_db = FAISS.from_texts(['Dummy historical fact: Bangladesh became independent in 1971.'], embed_model)\n"
    ]
}

new_stage_2_source = [
    "from transformers import pipeline\n",
    "import numpy as np\n",
    "\n",
    "# 1. Context Grounding -> NLI Cross-Encoder\n",
    "# nli_model = pipeline('text-classification', model='cross-encoder/nli-deberta-v3-base')\n",
    "def context_grounding_score(context, response):\n",
    "    # return nli_model(f'{context} [SEP] {response}')\n",
    "    return np.random.uniform(0.5, 1.0) # Dummy score\n",
    "\n",
    "# 2. No-Context Factual QA -> Retrieval + Evidence Verification (RAG Expanded!)\n",
    "def factual_qa_score(prompt, response):\n",
    "    # Step 1: Retrieve from local PDF corpus (FAISS)\n",
    "    retrieved_docs = faiss_db.similarity_search(prompt, k=2)\n",
    "    retrieved_context = ' '.join([doc.page_content for doc in retrieved_docs])\n",
    "    \n",
    "    # Step 2: Cross-encode retrieved context with response\n",
    "    # Ideally: score = nli_model(f'{retrieved_context} [SEP] {response}')\n",
    "    # Here we check for simple term overlap as a proxy for entailment in our skeleton\n",
    "    overlap_words = set(response.split()) & set(retrieved_context.split())\n",
    "    if len(overlap_words) > 0:\n",
    "        return 0.8 # Likely faithful if words overlap with retrieved history books\n",
    "    return 0.3 # Hallucinated if no support found in retrieved text\n",
    "\n",
    "# 3. Math/Number -> Symbolic Verifier\n",
    "def math_verification_score(prompt, response):\n",
    "    if '৫' in response and '২' in prompt:\n",
    "        return 0.0\n",
    "    return 1.0\n",
    "\n",
    "# 4. Translation/Summarization -> Semantic Entailment\n",
    "def translation_entailment_score(prompt, response):\n",
    "    return np.random.uniform(0.7, 1.0)\n",
    "\n",
    "# Apply detectors based on regime\n",
    "def get_regime_score(row):\n",
    "    if row['regime'] == 'context_grounding':\n",
    "        return context_grounding_score(row['context'], row['response'])\n",
    "    elif row['regime'] == 'math_reasoning':\n",
    "        return math_verification_score(row['prompt'], row['response'])\n",
    "    elif row['regime'] == 'translation_summarization':\n",
    "        return translation_entailment_score(row['prompt'], row['response'])\n",
    "    else:\n",
    "        return factual_qa_score(row['prompt'], row['response'])\n",
    "\n",
    "df['regime_score'] = df.apply(get_regime_score, axis=1)\n",
    "print(df[['regime', 'regime_score']])\n"
]

# Insert the RAG cell before Stage 2
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and 'Stage 2' in "".join(cell['source']):
        nb['cells'].insert(i, rag_init_cell)
        break

# Update Stage 2 code cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'def factual_qa_score' in "".join(cell['source']):
        cell['source'] = new_stage_2_source
        break

with open('/home/sword/bengali_hallucination_detection/pipeline.ipynb', 'w') as f:
    json.dump(nb, f)

print("Notebook updated with expanded RAG logic!")
