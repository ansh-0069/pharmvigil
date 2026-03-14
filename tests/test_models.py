"""
Tests for the data pipeline, feature engineering, and model training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestPreprocessing:
    """Tests for src.preprocessing.clean_data."""

    def test_handle_missing_values(self, sample_raw_df: pd.DataFrame):
        from src.preprocessing.clean_data import handle_missing_values

        df = sample_raw_df.copy()
        df.loc[0, "drug_name"] = None  # introduce missing critical field
        result = handle_missing_values(df)
        assert result["drug_name"].isnull().sum() == 0
        assert len(result) <= len(df)  # null row should be dropped or imputed

    def test_normalize_drugs(self, sample_raw_df: pd.DataFrame):
        from src.preprocessing.clean_data import normalize_drugs

        df = sample_raw_df.copy()
        df.loc[0, "drug_name"] = "  WARFARIN Sodium  "
        result = normalize_drugs(df)
        assert result.loc[0, "drug_name"] == "warfarin"

    def test_standardize_events(self, sample_raw_df: pd.DataFrame):
        from src.preprocessing.clean_data import standardize_events

        df = sample_raw_df.copy()
        df.loc[0, "adverse_event"] = "Heart Attack"
        result = standardize_events(df)
        assert result.loc[0, "adverse_event"] == "myocardial infarction"

    def test_full_preprocess(self, sample_raw_df: pd.DataFrame):
        from src.preprocessing.clean_data import preprocess

        result = preprocess(sample_raw_df.copy())
        assert len(result) > 0
        assert "drug_name" in result.columns


class TestFeatureEngineering:
    """Tests for src.features.feature_engineering."""

    def test_feature_matrix_shape(self, sample_features_df: pd.DataFrame):
        assert len(sample_features_df) > 0
        assert "prr" in sample_features_df.columns
        assert "ror" in sample_features_df.columns
        assert "co_occurrence_count" in sample_features_df.columns

    def test_prr_values_non_negative(self, sample_features_df: pd.DataFrame):
        assert (sample_features_df["prr"] >= 0).all()

    def test_drug_frequency_consistency(self, sample_features_df: pd.DataFrame):
        # Each drug should have frequency >= co-occurrence
        assert (
            sample_features_df["drug_frequency"]
            >= sample_features_df["co_occurrence_count"]
        ).all()

    def test_time_trend_exists(self, sample_features_df: pd.DataFrame):
        assert "time_trend_slope" in sample_features_df.columns


class TestModelTraining:
    """Tests for src.models.train_model."""

    def test_train_pipeline_produces_scores(self, sample_raw_df: pd.DataFrame):
        from src.models.train_model import train_pipeline

        result = train_pipeline(df=sample_raw_df.copy())
        assert len(result) > 0
        assert "risk_score" in result.columns
        assert "signal_strength" in result.columns
        assert "alert_level" in result.columns

    def test_risk_scores_in_range(self, sample_raw_df: pd.DataFrame):
        from src.models.train_model import train_pipeline

        result = train_pipeline(df=sample_raw_df.copy())
        assert result["risk_score"].between(0, 1).all(), "risk_score out of [0,1]"

    def test_alert_levels_valid(self, sample_raw_df: pd.DataFrame):
        from src.models.train_model import train_pipeline

        result = train_pipeline(df=sample_raw_df.copy())
        valid = {"low", "medium", "high", "critical"}
        assert set(result["alert_level"].unique()).issubset(valid)


class TestPrediction:
    """Tests for src.models.predict."""

    def test_predict_returns_result(self, sample_raw_df: pd.DataFrame):
        from src.models.train_model import train_pipeline

        # First ensure models exist
        train_pipeline(df=sample_raw_df.copy())

        from src.models.predict import SignalPredictor

        predictor = SignalPredictor()
        result = predictor.predict("warfarin", "thrombocytopenia")

        assert result.drug_name == "warfarin"
        assert result.adverse_event == "thrombocytopenia"
        assert 0 <= result.risk_score <= 1

    def test_top_signals(self, sample_raw_df: pd.DataFrame):
        from src.models.train_model import train_pipeline

        train_pipeline(df=sample_raw_df.copy())

        from src.models.predict import SignalPredictor

        predictor = SignalPredictor()
        signals = predictor.top_signals(n=5)
        assert len(signals) <= 5
        if signals:
            assert "drug_name" in signals[0]
            assert "risk_score" in signals[0]
