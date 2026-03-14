#!/usr/bin/env bash
# ─────────────────────────────────────────────────────
# Run the full training pipeline
# ─────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "══════════════════════════════════════════════════"
echo "  AI Pharmacovigilance — Training Pipeline"
echo "══════════════════════════════════════════════════"

# 1. Generate synthetic data (if not present)
if [ ! -f "data/raw/adverse_events.csv" ]; then
    echo "→ Generating synthetic FAERS data …"
    python scripts/generate_faers_data.py
fi

# 2. Run training
echo "→ Running model training …"
python -m src.models.train_model

echo ""
echo "✅ Training complete. Models saved in models/"
echo "   Signal scores saved in data/processed/signal_scores.csv"
