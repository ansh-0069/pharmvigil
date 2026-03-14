"""
Signal detection model — training pipeline.

Combines:
  1. Isolation Forest for anomaly / novelty scoring
  2. XGBoost classifier for signal classification

Outputs per drug-event pair:
  - risk_score      (0-1 continuous)
  - signal_strength (0-1 continuous)
  - alert_level     (low / medium / high / critical)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

from src.features.feature_engineering import build_feature_matrix, save_features
from src.ingestion.data_loader import load_all_faers
from src.preprocessing.clean_data import preprocess, save_processed
from src.utils.helpers import CONFIG


MODEL_DIR = Path(CONFIG.get("model", {}).get("artifact_dir", "models"))
FEATURE_COLS = [
    "co_occurrence_count",
    "drug_frequency",
    "event_frequency",
    "prr",
    "ror",
    "log_prr",
    "log_ror",
    "drug_event_ratio",
    "time_trend_slope",
]


# ── Alert-level assignment ───────────────────────────
def _assign_alert_level(score: float) -> str:
    thresholds = CONFIG.get("signal_detection", {}).get(
        "risk_thresholds", {"low": 0.3, "medium": 0.6, "high": 0.8}
    )
    if score >= thresholds.get("high", 0.8):
        return "critical"
    if score >= thresholds.get("medium", 0.6):
        return "high"
    if score >= thresholds.get("low", 0.3):
        return "medium"
    return "low"


def prepare_data(features: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Select and scale feature columns."""
    available = [c for c in FEATURE_COLS if c in features.columns]
    X = features[available].fillna(0).copy()
    return features, X.values


def train_isolation_forest(X: np.ndarray) -> Tuple[IsolationForest, np.ndarray]:
    """Fit Isolation Forest and return anomaly scores (higher = more anomalous)."""
    iso_cfg = CONFIG.get("model", {}).get("isolation_forest", {})
    iso = IsolationForest(
        n_estimators=int(iso_cfg.get("n_estimators", 200)),
        contamination=float(iso_cfg.get("contamination", 0.05)),
        random_state=int(iso_cfg.get("random_state", 42)),
        n_jobs=-1,
    )
    iso.fit(X)

    # decision_function: the lower, the more anomalous → invert & scale to [0,1]
    raw_scores = -iso.decision_function(X)
    scaler = MinMaxScaler()
    anomaly_scores = scaler.fit_transform(raw_scores.reshape(-1, 1)).ravel()

    return iso, anomaly_scores


def train_xgboost(X: np.ndarray, labels: np.ndarray) -> XGBClassifier:
    """Train XGBoost classifier on pseudo-labels derived from anomaly scores."""
    xgb_cfg = CONFIG.get("model", {}).get("xgboost", {})
    clf = XGBClassifier(
        n_estimators=int(xgb_cfg.get("n_estimators", 300)),
        max_depth=int(xgb_cfg.get("max_depth", 6)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.1)),
        random_state=int(xgb_cfg.get("random_state", 42)),
        use_label_encoder=False,
        eval_metric="logloss",
    )
    clf.fit(X, labels)
    return clf


def train_pipeline(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    End-to-end training pipeline.

    1. Load & preprocess data (if not provided)
    2. Build feature matrix
    3. Train Isolation Forest → anomaly scores
    4. Generate pseudo-labels → train XGBoost
    5. Produce risk_score, signal_strength, alert_level
    6. Save models & scored features

    Returns the scored DataFrame.
    """
    # ── Step 1: Data ──────────────────────────────────
    if df is None:
        raw = load_all_faers()
        if raw.empty:
            logger.error("No data found. Generate FAERS data first: python scripts/generate_faers_data.py")
            return pd.DataFrame()
        df = preprocess(raw)
        save_processed(df)

    # ── Step 2: Features ──────────────────────────────
    features = build_feature_matrix(df)
    save_features(features)

    # ── Step 3: Isolation Forest ──────────────────────
    logger.info("Training Isolation Forest …")
    meta, X = prepare_data(features)
    iso_model, anomaly_scores = train_isolation_forest(X)

    # ── Step 4: Pseudo-labels → XGBoost ───────────────
    threshold_75 = np.percentile(anomaly_scores, 75)
    pseudo_labels = (anomaly_scores >= threshold_75).astype(int)
    logger.info(
        f"  Pseudo-label distribution: {pseudo_labels.sum()} signals / {len(pseudo_labels)} total"
    )

    logger.info("Training XGBoost classifier …")
    xgb_model = train_xgboost(X, pseudo_labels)

    # ── Step 5: Score ─────────────────────────────────
    xgb_proba = xgb_model.predict_proba(X)[:, 1]

    features = features.copy()
    features["risk_score"] = np.round(
        0.4 * anomaly_scores + 0.6 * xgb_proba, 4
    )
    features["signal_strength"] = np.round(xgb_proba, 4)
    features["alert_level"] = features["risk_score"].apply(_assign_alert_level)

    # ── Step 6: Save ──────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(iso_model, MODEL_DIR / "isolation_forest.joblib")
    joblib.dump(xgb_model, MODEL_DIR / "xgboost_classifier.joblib")

    # Save the scaler fit on feature matrix for inference
    scaler = MinMaxScaler().fit(X)
    joblib.dump(scaler, MODEL_DIR / "feature_scaler.joblib")

    # Save scored results
    scored_path = Path("data/processed/signal_scores.csv")
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(scored_path, index=False)
    logger.info(f"Saved signal scores → {scored_path}")

    # Summary
    alert_counts = features["alert_level"].value_counts()
    logger.success(f"Training complete — {len(features):,} pairs scored")
    logger.info(f"  Alert distribution:\n{alert_counts.to_string()}")

    return features


# ── CLI entry-point ──────────────────────────────────
if __name__ == "__main__":
    from src.utils.helpers import setup_logging

    setup_logging("INFO")
    result = train_pipeline()
    if not result.empty:
        print(f"\nTop 10 riskiest drug-event pairs:")
        top = result.nlargest(10, "risk_score")[
            ["drug_name", "adverse_event", "risk_score", "signal_strength", "alert_level"]
        ]
        print(top.to_string(index=False))
