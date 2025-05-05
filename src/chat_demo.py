import streamlit as st
import os, json
import pandas as pd
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from difflib import SequenceMatcher
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load metrics
with open("data/processed/metrics.json") as f:
    metrics = json.load(f)
df_metrics = pd.DataFrame(metrics)

# Sidebar: Trust Summary
st.sidebar.title("📊 AI Trust Score Summary")
avg_score = df_metrics["ai_trust_score"].mean()
st.sidebar.metric("Avg Trust Score", f"{avg_score:.2f}")
st.sidebar.caption("🟢 ≥ 0.75 = High | 🟡 ≥ 0.5 = Medium | 🔴 < 0.5 = Low")

# Show per-document scores
st.sidebar.subheader("Trust Score per Document")
chart_data = df_metrics[["file", "ai_trust_score"]].set_index("file")
st.sidebar.bar_chart(chart_data)

# Split docs into AI-ready and Non-AI-ready
ai_ready_files = df_metrics[df_metrics["ai_trust_score"] >= 0.75]["file"].tolist()
non_ai_ready_files = df_metrics[df_metrics["ai_trust_score"] < 0.75]["file"].tolist()

# Load FAISS indices
ai_index_path = "faiss_index/ai_ready"
non_index_path = "faiss_index/non_ai_ready"

ai_vectorstore = FAISS.load_local(ai_index_path, embedding_model, allow_dangerous_deserialization=True)
non_vectorstore = FAISS.load_local(non_index_path, embedding_model, allow_dangerous_deserialization=True)

ai_qa = RetrievalQA.from_chain_type(
    llm=HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        temperature=0.3,
        max_new_tokens=512
    ),
    retriever=ai_vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

non_ai_qa = RetrievalQA.from_chain_type(
    llm=HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        temperature=0.3,
        max_new_tokens=512
    ),
    retriever=non_vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# UI
st.title("🧠 AI Readiness Chatbot: Compare Responses")
query = st.text_input("Enter your question to compare AI-ready vs non-AI-ready responses:")

if query:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟢 AI-Ready Answer")
        try:
            ai_response = ai_qa({"query": query})
            st.write(ai_response["result"])
            st.markdown("**Sources:**")
            for doc in ai_response["source_documents"]:
                base_file = doc.metadata["source"].split("_")[0] + ".txt"
                score = next((m["ai_trust_score"] for m in metrics if m["file"] == base_file), None)
                badge = "🟢" if score and score >= 0.75 else "🟡" if score and score >= 0.5 else "🔴"
                st.markdown(f"- `{doc.metadata['source']}` — {badge} Trust Score: **{score:.2f}**" if score else f"- `{doc.metadata['source']}`")
        except Exception as e:
            st.error(f"Error in AI-ready QA: {e}")

    with col2:
        st.subheader("🔴 Non-AI-Ready Answer")
        try:
            non_response = non_ai_qa({"query": query})
            st.write(non_response["result"])
            st.markdown("**Sources:**")
            for doc in non_response["source_documents"]:
                base_file = doc.metadata["source"].split("_")[0] + ".txt"
                score = next((m["ai_trust_score"] for m in metrics if m["file"] == base_file), None)
                badge = "🟢" if score and score >= 0.75 else "🟡" if score and score >= 0.5 else "🔴"
                st.markdown(f"- `{doc.metadata['source']}` — {badge} Trust Score: **{score:.2f}**" if score else f"- `{doc.metadata['source']}`")
        except Exception as e:
            st.error(f"Error in non-AI-ready QA: {e}")

    # Response similarity
    try:
        similarity = SequenceMatcher(None, ai_response["result"], non_response["result"]).ratio()
        st.markdown("---")
        st.markdown(f"### 🔍 Response Similarity Score: `{similarity:.2f}`")
        if similarity < 0.6:
            st.error("Large difference — AI-Ready data provided significantly better guidance.")
        elif similarity < 0.85:
            st.warning("Moderate difference — noticeable variation in content.")
        else:
            st.success("Minimal difference — both data sources performed similarly.")
    except:
        st.warning("Could not compute similarity between answers.")

