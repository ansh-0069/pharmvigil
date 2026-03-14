#!/usr/bin/env bash
# ─────────────────────────────────────────────────────
# Run the full ETL pipeline (without Airflow)
# ─────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "══════════════════════════════════════════════════"
echo "  AI Pharmacovigilance — ETL Pipeline"
echo "══════════════════════════════════════════════════"

# 1. Generate data
echo "→ Step 1: Generating synthetic FAERS data …"
python scripts/generate_faers_data.py

# 2. Train model (includes ingestion + preprocessing + feature engineering)
echo "→ Step 2: Running training pipeline …"
python -m src.models.train_model

# 3. Launch API
echo "→ Step 3: Starting API server …"
echo "   Run: uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "✅ Pipeline complete!"
