import streamlit as st
import json
import pandas as pd

st.set_page_config(page_title="📊 AI Readiness Report", layout="centered")
st.title("📊 AI Readiness Dashboard")

# Load processed metrics
with open("data/processed/metrics.json") as f:
    metrics = json.load(f)

df = pd.DataFrame(metrics)

# Classify each entry by document type
def classify_file(file):
    return "AI-Ready" if "cymbalta" in file.lower() else "Non-AI-Ready"

df["type"] = df["file"].apply(classify_file)

# Sidebar: file names
st.sidebar.header("📂 Document Breakdown")
for doc_type in df["type"].unique():
    st.sidebar.subheader(doc_type)
    st.sidebar.write(df[df["type"] == doc_type]["file"].unique().tolist())

# Compute average metrics
def avg_scores(data):
    return {
        "Completeness": data["completeness"].mean(),
        "Accuracy": data["accuracy"].mean(),
        "Secure": data["secure"].mean(),
        "Quality": data["quality"].mean(),
        "Timeliness": data["timeliness"].mean(),
        "Token Count": data["token_count"].mean()
    }

ai_ready = df[df["type"] == "AI-Ready"]
non_ai_ready = df[df["type"] == "Non-AI-Ready"]

ai_metrics = avg_scores(ai_ready)
non_ai_metrics = avg_scores(non_ai_ready)

# Display comparison
st.markdown("### 🆚 Metric Breakdown: AI-Ready vs Non-AI-Ready")
comparison_df = pd.DataFrame({
    "Metric": list(ai_metrics.keys()),
    "AI-Ready": list(ai_metrics.values()),
    "Non-AI-Ready": list(non_ai_metrics.values())
})

st.dataframe(comparison_df, use_container_width=True)

