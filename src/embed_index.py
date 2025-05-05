import os
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_docs_by_prefix(prefix):
    docs = []
    for fname in os.listdir("data/processed"):
        if fname.endswith(".txt") and fname.startswith(prefix):
            with open(f"data/processed/{fname}", "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:  # skip empty
                    docs.append(Document(page_content=content, metadata={"source": fname}))
    return docs

print("📦 Creating FAISS index...")

# Load and validate documents
ai_docs = load_docs_by_prefix("cymbalta")
non_ai_docs = load_docs_by_prefix("noisy_drug_info")

if not ai_docs:
    raise ValueError("❌ No AI-ready documents found in 'data/processed' with prefix 'cymbalta'")
if not non_ai_docs:
    raise ValueError("❌ No Non-AI-ready documents found in 'data/processed' with prefix 'noisy_drug_info'")

# Embed and store
FAISS.from_documents(ai_docs, embedding_model).save_local("faiss_index/ai_ready")
FAISS.from_documents(non_ai_docs, embedding_model).save_local("faiss_index/non_ai_ready")

print("✅ FAISS indexes saved to 'faiss_index/'")

