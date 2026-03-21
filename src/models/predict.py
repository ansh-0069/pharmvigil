"""
Prediction / inference module — load trained models and score new drug-event pairs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.utils.helpers import CONFIG


MODEL_DIR = Path(CONFIG.get("model", {}).get("artifact_dir", "models"))
SCORED_PATH = Path("data/processed/signal_scores.csv")

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


@dataclass
class PredictionResult:
    drug_name: str
    adverse_event: str
    risk_score: float
    signal_strength: float
    alert_level: str


def _assign_alert_level(score: float) -> str:
    thresholds = CONFIG.get("signal_detection", {}).get(
        "risk_thresholds", {"low": 0.3, "medium": 0.6, "high": 0.8}
    )
    if score >= float(thresholds.get("high", 0.8)):
        return "critical"
    if score >= float(thresholds.get("medium", 0.6)):
        return "high"
    if score >= float(thresholds.get("low", 0.3)):
        return "medium"
    return "low"


class SignalPredictor:
    """Load trained models and predict risk for drug-event pairs."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or MODEL_DIR
        self.iso_model = None
        self.xgb_model = None
        self.scaler = None
        self.scored_cache: Optional[pd.DataFrame] = None
        self._load_models()
        self._load_scored_cache()

    def _load_models(self) -> None:
        """Load serialised model artifacts."""
        iso_path = self.model_dir / "isolation_forest.joblib"
        xgb_path = self.model_dir / "xgboost_classifier.joblib"
        scaler_path = self.model_dir / "feature_scaler.joblib"

        if iso_path.exists():
            try:
                self.iso_model = joblib.load(iso_path)
                logger.info("Loaded Isolation Forest model")
            except Exception as exc:
                logger.warning(f"Could not load Isolation Forest model: {exc}")
        if xgb_path.exists():
            try:
                self.xgb_model = joblib.load(xgb_path)
                logger.info("Loaded XGBoost classifier")
            except Exception as exc:
                logger.warning(f"Could not load XGBoost model: {exc}")
        if scaler_path.exists():
            try:
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
            except Exception as exc:
                logger.warning(f"Could not load feature scaler: {exc}")

    def _load_scored_cache(self) -> None:
        """Load pre-computed signal scores for fast lookup."""
        if SCORED_PATH.exists():
            try:
                self.scored_cache = pd.read_csv(SCORED_PATH)
                logger.info(f"Loaded {len(self.scored_cache):,} cached signal scores")
            except Exception as exc:
                logger.warning(f"Could not load cached signal scores: {exc}")

    def predict(self, drug_name: str, adverse_event: str) -> PredictionResult:
        """
        Predict risk score for a drug-event pair.

        First checks the pre-computed cache; if not found, runs inference
        through the loaded models.
        """
        drug_name = drug_name.lower().strip()
        adverse_event = adverse_event.lower().strip()

        # ── Cache lookup ──────────────────────────────
        if self.scored_cache is not None:
            match = self.scored_cache[
                (self.scored_cache["drug_name"].str.lower() == drug_name)
                & (self.scored_cache["adverse_event"].str.lower() == adverse_event)
            ]
            if not match.empty:
                row = match.iloc[0]
                risk_val = float(row.get("risk_score", 0.0))
                signal_val = float(row.get("signal_strength", 0.0))
                alert_val = str(row.get("alert_level", _assign_alert_level(risk_val)))
                return PredictionResult(
                    drug_name=drug_name,
                    adverse_event=adverse_event,
                    risk_score=round(risk_val, 4),
                    signal_strength=round(signal_val, 4),
                    alert_level=alert_val,
                )

        # ── Model inference ───────────────────────────
        if self.iso_model is None or self.xgb_model is None:
            logger.warning("Models not loaded — returning default scores")
            return PredictionResult(
                drug_name=drug_name,
                adverse_event=adverse_event,
                risk_score=0.0,
                signal_strength=0.0,
                alert_level="low",
            )

        # Build a minimal feature vector (zeros for unknown pairs)
        X = np.zeros((1, len(FEATURE_COLS)))

        try:
            if self.scaler is not None:
                X = self.scaler.transform(X)

            iso_raw = -self.iso_model.decision_function(X)
            iso_score = float(np.clip(iso_raw, 0, 1))

            xgb_proba = float(self.xgb_model.predict_proba(X)[:, 1])

            risk = round(0.4 * iso_score + 0.6 * xgb_proba, 4)
            return PredictionResult(
                drug_name=drug_name,
                adverse_event=adverse_event,
                risk_score=risk,
                signal_strength=round(xgb_proba, 4),
                alert_level=_assign_alert_level(risk),
            )
        except Exception as exc:
            # Keep /predict resilient even if model artifacts are shape-incompatible.
            logger.exception(f"Prediction inference failed for {drug_name}/{adverse_event}: {exc}")
            return PredictionResult(
                drug_name=drug_name,
                adverse_event=adverse_event,
                risk_score=0.0,
                signal_strength=0.0,
                alert_level="low",
            )

    def top_signals(self, n: int = 50) -> List[Dict]:
        """Return the top-N highest-risk drug-event pairs from the cache."""
        if self.scored_cache is None or self.scored_cache.empty:
            return []

        top = self.scored_cache.nlargest(n, "risk_score")
        cols = ["drug_name", "adverse_event", "risk_score", "signal_strength", "alert_level"]
        available = [c for c in cols if c in top.columns]
        return top[available].to_dict(orient="records")


# ── Module-level singleton ───────────────────────────
_predictor: Optional[SignalPredictor] = None


def get_predictor() -> SignalPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SignalPredictor()
    return _predictor
