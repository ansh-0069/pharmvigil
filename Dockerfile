# ─── Stage 1: Builder ────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
RUN pip install --no-cache-dir --prefix=/install spacy && \
    python -m spacy download en_core_web_sm

# ─── Stage 2: Runtime ────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="pharmacovigilance-team"
LABEL description="AI Pharmacovigilance Intelligence Platform"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create directories
RUN mkdir -p data/raw data/processed models logs

# Expose API and dashboard ports
EXPOSE 8000 8501

# Default: run the API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
