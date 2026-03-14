"""
Data ingestion — load FAERS-format CSVs and upsert into PostgreSQL.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from sqlalchemy import text

from src.database.db import AdverseEvent, engine, get_session, init_db
from src.utils.helpers import CONFIG


RAW_DIR = Path(CONFIG.get("data", {}).get("raw_dir", "data/raw"))


def load_csv(filename: str, directory: Optional[Path] = None) -> pd.DataFrame:
    """Read a single CSV from the raw data directory."""
    directory = directory or RAW_DIR
    filepath = directory / filename
    logger.info(f"Loading CSV: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    logger.info(f"  → {len(df):,} rows, {len(df.columns)} columns")
    return df


def load_all_faers(directory: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and merge FAERS CSV files into a single DataFrame.

    Expected files:
      - adverse_events.csv  (core reports)
      - drug_info.csv       (drug details, optional)
      - demographics.csv    (patient demographics, optional)
    """
    directory = directory or RAW_DIR
    ae_path = directory / "adverse_events.csv"

    if not ae_path.exists():
        logger.warning(f"No adverse_events.csv found in {directory}")
        return pd.DataFrame()

    df = load_csv("adverse_events.csv", directory)

    # Merge drug info if available
    drug_path = directory / "drug_info.csv"
    if drug_path.exists():
        drug_df = load_csv("drug_info.csv", directory)
        merge_col = "report_id" if "report_id" in drug_df.columns else None
        if merge_col and merge_col in df.columns:
            df = df.merge(drug_df, on=merge_col, how="left", suffixes=("", "_drug"))
            logger.info(f"  Merged drug_info → {len(df):,} rows")

    # Merge demographics if available
    demo_path = directory / "demographics.csv"
    if demo_path.exists():
        demo_df = load_csv("demographics.csv", directory)
        merge_col = "report_id" if "report_id" in demo_df.columns else None
        if merge_col and merge_col in df.columns:
            df = df.merge(demo_df, on=merge_col, how="left", suffixes=("", "_demo"))
            logger.info(f"  Merged demographics → {len(df):,} rows")

    return df


def ingest_to_db(df: pd.DataFrame, batch_size: int = 1000) -> int:
    """
    Insert a DataFrame of adverse event records into PostgreSQL.
    Returns the number of rows inserted.
    """
    init_db()

    columns = [c.name for c in AdverseEvent.__table__.columns if c.name != "id"]
    df_to_insert = df[[c for c in columns if c in df.columns]].copy()

    total = 0
    with get_session() as session:
        for start in range(0, len(df_to_insert), batch_size):
            batch = df_to_insert.iloc[start : start + batch_size]
            records = batch.to_dict(orient="records")
            session.bulk_insert_mappings(AdverseEvent, records)
            session.flush()
            total += len(records)
            logger.info(f"  Inserted batch {start // batch_size + 1} ({total:,} rows)")

    logger.success(f"Ingested {total:,} adverse event records into database")
    return total


def get_record_count() -> int:
    """Return the current count of adverse_events rows."""
    with engine.connect() as conn:
        try:
            result = conn.execute(text("SELECT COUNT(*) FROM adverse_events"))
            return result.scalar() or 0
        except Exception:
            return 0
