.PHONY: install test lint train serve dashboard generate-data clean

# ── Setup ─────────────────────────────────────────────
install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

# ── Data ──────────────────────────────────────────────
generate-data:
	python scripts/generate_faers_data.py

# ── Training ──────────────────────────────────────────
train:
	python -m src.models.train_model

# ── Serve ─────────────────────────────────────────────
serve:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard/streamlit_app.py --server.port 8501

# ── Quality ───────────────────────────────────────────
test:
	python -m pytest tests/ -v --tb=short

lint:
	python -m py_compile src/api/app.py
	python -m py_compile src/models/train_model.py
	python -m py_compile src/models/predict.py
	python -m py_compile src/nlp/entity_extraction.py
	python -m py_compile src/features/feature_engineering.py

# ── Cleanup ───────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov
