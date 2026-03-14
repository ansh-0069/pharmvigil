"""
Data preprocessing — cleaning, normalisation, and standardisation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.helpers import CONFIG, normalize_drug_name, normalize_event_term


PROCESSED_DIR = Path(CONFIG.get("data", {}).get("processed_dir", "data/processed"))

# ── MedDRA-style event mapping (simplified) ─────────
EVENT_SYNONYM_MAP: dict[str, str] = {
    "heart attack": "myocardial infarction",
    "mi": "myocardial infarction",
    "stroke": "cerebrovascular accident",
    "cva": "cerebrovascular accident",
    "liver failure": "hepatic failure",
    "kidney failure": "renal failure",
    "rash": "dermatitis",
    "skin rash": "dermatitis",
    "stomach pain": "abdominal pain",
    "vomiting": "emesis",
    "high blood pressure": "hypertension",
    "low blood pressure": "hypotension",
    "shortness of breath": "dyspnea",
    "breathing difficulty": "dyspnea",
    "chest pain": "angina pectoris",
    "blurred vision": "visual impairment",
    "joint pain": "arthralgia",
    "muscle pain": "myalgia",
    "blood clot": "thrombosis",
    "allergic reaction": "hypersensitivity",
    "swelling": "edema",
    "weight gain": "weight increased",
    "weight loss": "weight decreased",
    "hair loss": "alopecia",
    "sleeplessness": "insomnia",
    "tiredness": "fatigue",
    "confusion": "confusional state",
    "memory loss": "amnesia",
    "seizure": "convulsion",
    "infection": "infectious disorder",
}


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute or drop rows/columns with excessive missing data."""
    logger.info("Handling missing values …")
    initial = len(df)

    # Drop columns that are >80 % null
    null_pct = df.isnull().mean()
    drop_cols = null_pct[null_pct > 0.8].index.tolist()
    if drop_cols:
        logger.info(f"  Dropping columns (>80 % null): {drop_cols}")
        df = df.drop(columns=drop_cols)

    # Impute numeric columns with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Impute categorical columns with "unknown"
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("unknown")

    # Drop rows where critical fields are still missing
    critical = ["drug_name", "adverse_event"]
    existing_critical = [c for c in critical if c in df.columns]
    if existing_critical:
        df = df.dropna(subset=existing_critical)

    logger.info(f"  Rows: {initial:,} → {len(df):,}")
    return df.reset_index(drop=True)


def normalize_drugs(df: pd.DataFrame, col: str = "drug_name") -> pd.DataFrame:
    """Normalise drug names in the given column."""
    if col not in df.columns:
        return df
    logger.info(f"Normalising drug names in '{col}' …")
    df[col] = df[col].apply(normalize_drug_name)
    return df


def standardize_events(df: pd.DataFrame, col: str = "adverse_event") -> pd.DataFrame:
    """Map adverse event terms to standardised MedDRA-style preferred terms."""
    if col not in df.columns:
        return df
    logger.info(f"Standardising adverse event terms in '{col}' …")
    df[col] = df[col].apply(normalize_event_term)
    df[col] = df[col].replace(EVENT_SYNONYM_MAP)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline."""
    df = handle_missing_values(df)
    df = normalize_drugs(df)
    df = standardize_events(df)

    # Parse dates
    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")

    logger.success(f"Preprocessing complete — {len(df):,} clean records")
    return df


def save_processed(df: pd.DataFrame, filename: str = "cleaned_reports.csv") -> Path:
    """Persist the cleaned DataFrame to disk."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / filename
    df.to_csv(out, index=False)
    logger.info(f"Saved processed data → {out}")
    return out
