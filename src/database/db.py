"""
Database layer — SQLAlchemy engine, session factory, and ORM models.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.utils.helpers import CONFIG

# ── Connection URL ───────────────────────────────────
_DB_URL: str = CONFIG.get("database", {}).get(
    "url",
    "postgresql://{user}:{pw}@{host}:{port}/{db}".format(
        user=os.getenv("DB_USER", "pharma_user"),
        pw=os.getenv("DB_PASSWORD", "pharma_pass"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        db=os.getenv("DB_NAME", "pharmacovigilance"),
    ),
)

engine = create_engine(_DB_URL, pool_pre_ping=True, pool_size=10)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# ── ORM Models ───────────────────────────────────────
class AdverseEvent(Base):
    """Raw adverse-event report row."""

    __tablename__ = "adverse_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(64), index=True, nullable=False)
    patient_age = Column(Integer)
    patient_sex = Column(String(10))
    patient_weight = Column(Float)
    drug_name = Column(String(256), index=True)
    drug_dose = Column(String(128))
    route_of_admin = Column(String(64))
    indication = Column(String(256))
    adverse_event = Column(String(256), index=True)
    outcome = Column(String(64))
    severity = Column(String(32))
    report_date = Column(DateTime)
    country = Column(String(64))
    narrative = Column(Text)
    created_at = Column(DateTime, server_default=func.now())


class DrugEventPair(Base):
    """Aggregated drug-event pair with computed frequencies."""

    __tablename__ = "drug_event_pairs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    drug_name = Column(String(256), index=True, nullable=False)
    adverse_event = Column(String(256), index=True, nullable=False)
    co_occurrence_count = Column(Integer, default=0)
    drug_frequency = Column(Integer, default=0)
    event_frequency = Column(Integer, default=0)
    total_reports = Column(Integer, default=0)
    prr = Column(Float)  # Proportional Reporting Ratio
    ror = Column(Float)  # Reporting Odds Ratio
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class SignalScore(Base):
    """ML-generated signal detection scores."""

    __tablename__ = "signal_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    drug_name = Column(String(256), index=True, nullable=False)
    adverse_event = Column(String(256), index=True, nullable=False)
    risk_score = Column(Float, nullable=False)
    signal_strength = Column(Float, nullable=False)
    alert_level = Column(String(16), nullable=False)  # low / medium / high / critical
    model_version = Column(String(32))
    scored_at = Column(DateTime, server_default=func.now())


# ── Helpers ──────────────────────────────────────────
def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide a transactional session scope."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
