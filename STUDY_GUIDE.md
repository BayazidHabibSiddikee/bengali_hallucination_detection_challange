# 🧠 Study Guide: Advanced NLP & Hallucination Detection

If you want to deeply understand the code in your Kaggle pipeline, here are the 5 core concepts it uses, explained simply, along with links to study them further.

---

## 1. RAG (Retrieval-Augmented Generation) & TF-IDF
**How your pipeline uses it:** When a question is missing its "context", the pipeline searches Bengali Wikipedia and your Historical Books to find the answer before judging it. 
**How it works:** Instead of heavy neural networks, it uses **TF-IDF** (Term Frequency-Inverse Document Frequency). It calculates how rare words are. If a Wikipedia article shares very rare words with the question, it's considered a match.
- **Learn TF-IDF:** [TF-IDF Explained (FreeCodeCamp)](https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/)
- **Learn RAG:** [IBM's Guide to RAG](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)

## 2. Fine-Tuning Transformer Models (BanglaBERT)
**How your pipeline uses it:** It takes a pre-trained BERT model (which understands the Bengali language) and trains a "Classification Head" on top of it to classify text as Hallucination (0) or Faithful (1).
**How it works:** BERT reads text bidirectionally to understand context. During fine-tuning, we pass in 25,000 examples and slightly tweak the weights of the model using Backpropagation so it gets better at our specific binary classification task.
- **Learn BERT:** [The Illustrated BERT (Jay Alammar)](https://jalammar.github.io/illustrated-bert/)
- **Learn Fine-Tuning via HuggingFace:** [HuggingFace Fine-Tuning Tutorial](https://huggingface.co/docs/transformers/training)

## 3. LLM-as-a-Judge (Logit Extraction)
**How your pipeline uses it:** It uses TigerLLM-9B to judge hallucinations, but it *never asks it to generate text*. Generating text is slow.
**How it works:** Instead of generating text, we do a single "forward pass". We look at the final mathematical array (Logits) right before the LLM decides what word to type next. We extract the exact probability of the LLM wanting to say "1" vs "0". This is 100x faster than generating text and mathematically more accurate.
- **Learn Logits & Softmax:** [Understanding Neural Network Logits](https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax)
- **Learn LLM-as-a-Judge:** [Judging LLM-as-a-Judge (HuggingFace Paper)](https://huggingface.co/papers/2306.05685)

## 4. Focal Loss (Handling Imbalanced Data)
**How your pipeline uses it:** When training BanglaBERT, most of the augmented data is "easy" to solve. Focal Loss forces the model to ignore the easy ones and focus entirely on the examples it keeps getting wrong.
**How it works:** Standard Cross-Entropy loss treats all mistakes equally. Focal Loss applies a mathematical "decay" to confident predictions, so the gradient updates only care about the tricky edge cases.
- **Learn Focal Loss:** [Focal Loss for Dense Object Detection (Original Paper Explanation)](https://amaarora.github.io/2020/06/29/FocalLoss.html)

## 5. Model Ensembling & Powell Optimization
**How your pipeline uses it:** BanglaBERT and TigerLLM-9B will disagree sometimes. Which one should we trust?
**How it works:** The Powell Optimizer algorithm tests thousands of different percentage blends (e.g., 60% BERT, 40% TigerLLM) on the Validation set to find the exact mathematical blend that results in the highest F1 Score.
- **Learn Ensembling:** [Machine Learning Ensembles (Scikit-Learn)](https://scikit-learn.org/stable/modules/ensemble.html)
- **Learn Powell's Method:** [Powell's Optimization Method (SciPy)](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html)
