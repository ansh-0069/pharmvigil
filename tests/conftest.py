"""
Shared pytest fixtures for the pharmacovigilance test suite.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def sample_raw_df() -> pd.DataFrame:
    """Minimal raw adverse-event DataFrame for testing."""
    return pd.DataFrame(
        {
            "report_id": [f"TEST-{i:04d}" for i in range(1, 51)],
            "patient_age": [45, 62, 33, 70, 55] * 10,
            "patient_sex": ["male", "female"] * 25,
            "patient_weight": [75.0, 60.0, 90.5, 68.0, 82.0] * 10,
            "drug_name": (
                ["warfarin"] * 10
                + ["metformin"] * 10
                + ["ibuprofen"] * 10
                + ["pembrolizumab"] * 10
                + ["ciprofloxacin"] * 10
            ),
            "drug_dose": ["10 mg"] * 50,
            "route_of_admin": ["oral"] * 50,
            "indication": ["hypertension"] * 50,
            "adverse_event": (
                ["thrombocytopenia"] * 5
                + ["nausea"] * 5
                + ["hypoglycemia"] * 5
                + ["diarrhea"] * 5
                + ["headache"] * 5
                + ["hepatotoxicity"] * 5
                + ["fatigue"] * 5
                + ["seizure"] * 5
                + ["confusion"] * 5
                + ["rash"] * 5
            ),
            "outcome": ["recovered"] * 50,
            "severity": ["mild", "moderate", "severe", "mild", "moderate"] * 10,
            "report_date": pd.date_range("2023-01-01", periods=50, freq="W").strftime(
                "%Y-%m-%d"
            ),
            "country": ["US"] * 30 + ["UK"] * 10 + ["DE"] * 10,
            "narrative": [
                f"Patient reported adverse event after taking drug."
            ]
            * 50,
        }
    )


@pytest.fixture(scope="session")
def sample_features_df(sample_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix built from sample data."""
    from src.features.feature_engineering import build_feature_matrix

    return build_feature_matrix(sample_raw_df)
