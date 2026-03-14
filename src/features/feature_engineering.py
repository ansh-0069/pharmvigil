"""
Feature engineering — compute drug-event features for signal detection.

Features produced per drug-event pair:
  - drug_frequency        : how often the drug appears across all reports
  - event_frequency       : how often the event appears across all reports
  - co_occurrence_count   : how many reports contain both the drug AND the event
  - total_reports         : total report count in the dataset
  - prr                   : Proportional Reporting Ratio
  - ror                   : Reporting Odds Ratio
  - log_prr / log_ror     : log-transformed versions (better for ML)
  - time_trend_slope      : linear slope of monthly co-occurrence counts
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def compute_drug_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pharmacovigilance features for every unique (drug, event) pair.

    Parameters
    ----------
    df : DataFrame with at least ``drug_name`` and ``adverse_event`` columns.

    Returns
    -------
    DataFrame with one row per (drug_name, adverse_event) pair and feature columns.
    """
    required = {"drug_name", "adverse_event"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    logger.info("Computing drug-event features …")
    total_reports = len(df)

    # ── Per-drug and per-event counts ────────────────
    drug_counts = df["drug_name"].value_counts().rename("drug_frequency")
    event_counts = df["adverse_event"].value_counts().rename("event_frequency")

    # ── Co-occurrence counts ─────────────────────────
    pairs = (
        df.groupby(["drug_name", "adverse_event"])
        .size()
        .reset_index(name="co_occurrence_count")
    )

    pairs = pairs.merge(drug_counts, left_on="drug_name", right_index=True)
    pairs = pairs.merge(event_counts, left_on="adverse_event", right_index=True)
    pairs["total_reports"] = total_reports

    # ── PRR (Proportional Reporting Ratio) ───────────
    # PRR = (a / (a+b)) / (c / (c+d))
    # a = co-occurrence, b = drug reports without event
    # c = event reports without drug, d = reports without either
    a = pairs["co_occurrence_count"]
    b = pairs["drug_frequency"] - a
    c = pairs["event_frequency"] - a
    d = total_reports - a - b - c

    prr_num = a / (a + b).replace(0, np.nan)
    prr_den = c / (c + d).replace(0, np.nan)
    pairs["prr"] = (prr_num / prr_den.replace(0, np.nan)).fillna(0).round(4)

    # ── ROR (Reporting Odds Ratio) ───────────────────
    ror_num = (a * d).replace(0, np.nan)
    ror_den = (b * c).replace(0, np.nan)
    pairs["ror"] = (ror_num / ror_den).fillna(0).round(4)

    # ── Log transforms ───────────────────────────────
    pairs["log_prr"] = np.log1p(pairs["prr"]).round(4)
    pairs["log_ror"] = np.log1p(pairs["ror"]).round(4)

    # ── Relative frequency ───────────────────────────
    pairs["drug_event_ratio"] = (
        pairs["co_occurrence_count"] / pairs["drug_frequency"]
    ).round(4)

    logger.info(f"  Computed features for {len(pairs):,} drug-event pairs")
    return pairs


def compute_time_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly co-occurrence counts and a linear trend slope
    for each (drug, event) pair.
    """
    if "report_date" not in df.columns:
        logger.warning("No 'report_date' column — skipping time trends")
        return pd.DataFrame()

    df = df.copy()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.dropna(subset=["report_date"])
    df["year_month"] = df["report_date"].dt.to_period("M")

    monthly = (
        df.groupby(["drug_name", "adverse_event", "year_month"])
        .size()
        .reset_index(name="monthly_count")
    )

    # Linear slope per pair
    def _slope(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return 0.0
        x = np.arange(len(group), dtype=float)
        y = group["monthly_count"].values.astype(float)
        coeffs = np.polyfit(x, y, 1)
        return round(coeffs[0], 4)

    trends = (
        monthly.groupby(["drug_name", "adverse_event"])
        .apply(_slope, include_groups=False)
        .reset_index(name="time_trend_slope")
    )

    logger.info(f"  Computed time trends for {len(trends):,} pairs")
    return trends


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline: pair features + time trends."""
    logger.info("Building feature matrix …")

    features = compute_drug_event_features(df)

    trends = compute_time_trends(df)
    if not trends.empty:
        features = features.merge(
            trends, on=["drug_name", "adverse_event"], how="left"
        )
        features["time_trend_slope"] = features["time_trend_slope"].fillna(0)
    else:
        features["time_trend_slope"] = 0.0

    logger.success(f"Feature matrix: {features.shape[0]:,} pairs × {features.shape[1]} features")
    return features


def save_features(features: pd.DataFrame, path: str = "data/processed/features.csv") -> Path:
    """Persist the feature matrix."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)
    logger.info(f"Saved features → {out}")
    return out
