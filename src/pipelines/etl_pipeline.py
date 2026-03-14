"""
Airflow DAG — Pharmacovigilance ETL Pipeline.

Schedule: daily
Flow:  ingest → preprocess → feature_engineer → train_model → update_predictions
"""
from __future__ import annotations

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

# ── Task functions ───────────────────────────────────

def _ingest(**kwargs):
    """Load raw FAERS CSVs."""
    from src.ingestion.data_loader import load_all_faers
    df = load_all_faers()
    csv_path = "data/processed/_ingested_tmp.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _preprocess(**kwargs):
    """Clean and normalise."""
    import pandas as pd
    from src.preprocessing.clean_data import preprocess, save_processed

    ti = kwargs["ti"]
    csv_path = ti.xcom_pull(task_ids="ingest_data")
    df = pd.read_csv(csv_path)
    df = preprocess(df)
    out = save_processed(df)
    return str(out)


def _feature_engineer(**kwargs):
    """Build feature matrix."""
    import pandas as pd
    from src.features.feature_engineering import build_feature_matrix, save_features

    ti = kwargs["ti"]
    csv_path = ti.xcom_pull(task_ids="preprocess_data")
    df = pd.read_csv(csv_path)
    features = build_feature_matrix(df)
    out = save_features(features)
    return str(out)


def _train_model(**kwargs):
    """Train signal detection models."""
    import pandas as pd
    from src.models.train_model import train_pipeline

    ti = kwargs["ti"]
    csv_path = ti.xcom_pull(task_ids="preprocess_data")
    df = pd.read_csv(csv_path)
    train_pipeline(df)
    return "training_complete"


def _update_predictions(**kwargs):
    """Refresh the prediction cache."""
    from src.models.predict import get_predictor

    predictor = get_predictor()
    predictor._load_scored_cache()
    signals = predictor.top_signals(n=10)
    return f"Top signal: {signals[0] if signals else 'none'}"


# ── DAG Definition ───────────────────────────────────
if AIRFLOW_AVAILABLE:
    default_args = {
        "owner": "pharmacovigilance-team",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="pharmacovigilance_etl",
        default_args=default_args,
        description="End-to-end pharmacovigilance data pipeline",
        schedule_interval="@daily",
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["pharmacovigilance", "ml", "etl"],
    ) as dag:

        t_ingest = PythonOperator(
            task_id="ingest_data",
            python_callable=_ingest,
        )

        t_preprocess = PythonOperator(
            task_id="preprocess_data",
            python_callable=_preprocess,
        )

        t_features = PythonOperator(
            task_id="feature_engineer",
            python_callable=_feature_engineer,
        )

        t_train = PythonOperator(
            task_id="train_model",
            python_callable=_train_model,
        )

        t_predict = PythonOperator(
            task_id="update_predictions",
            python_callable=_update_predictions,
        )

        t_ingest >> t_preprocess >> t_features >> t_train >> t_predict
