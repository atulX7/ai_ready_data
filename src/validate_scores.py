import json
import os
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Load trust scores
with open("data/processed/metrics.json") as f:
    metrics = json.load(f)

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Collect embeddings
def get_embeddings(threshold, condition):
    texts = []
    for m in metrics:
        score = m["ai_trust_score"]
        if condition(score, threshold):
            fname = m["file"].replace(".pdf", ".txt")
            fpath = os.path.join("data/processed", fname)
            if os.path.exists(fpath):
                with open(fpath, encoding="utf-8") as f:
                    content = f.read()
                    if content:
                        texts.append(content)
    return embedding_model.embed_documents(texts)

ai_ready_embeddings = get_embeddings(0.75, lambda s, t: s >= t)
non_ready_embeddings = get_embeddings(0.75, lambda s, t: s < t)

def avg_similarity(vectors):
    if len(vectors) < 2:
        return 0.0
    sims = []
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            sims.append(cosine_similarity([vectors[i]], [vectors[j]])[0][0])
    return np.mean(sims)

print("🔍 Embedding Quality Validation:")
print(f"🟢 AI-Ready Avg Cosine Similarity: {avg_similarity(ai_ready_embeddings):.3f}")
print(f"🔴 Non-AI-Ready Avg Cosine Similarity: {avg_similarity(non_ready_embeddings):.3f}")

