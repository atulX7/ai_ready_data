# ----------- CONFIG -------------
PYTHON=python
SRC_DIR=src
PORT_CHAT=8501
PORT_REPORT=8502

# ----------- INGEST & PIPELINE -------------
ingest:
	@echo "📥 Running data ingestion..."
	$(PYTHON) $(SRC_DIR)/ingest.py

score:
	@echo "📊 Running AI Trust Score evaluation..."
	$(PYTHON) $(SRC_DIR)/score.py

embed:
	@echo "📦 Creating FAISS index..."
	$(PYTHON) $(SRC_DIR)/embed_index.py

validate:
	@echo "🔍 Running validation script..."
	$(PYTHON) $(SRC_DIR)/validate_scores.py

# ----------- STREAMLIT APPS -------------
chat:
	@echo "💬 Launching chatbot app on port $(PORT_CHAT)..."
	streamlit run $(SRC_DIR)/chat_demo.py --server.port $(PORT_CHAT)

report:
	@echo "📊 Launching AI Readiness dashboard on port $(PORT_REPORT)..."
	streamlit run $(SRC_DIR)/report.py --server.port $(PORT_REPORT)

# ----------- CLEANUP -------------
clean:
	@echo "🧹 Cleaning up generated files..."
	rm -rf faiss_index/ data/processed/*.txt data/processed/metrics.json

# ----------- ALL-IN-ONE -------------
all: ingest score embed validate


