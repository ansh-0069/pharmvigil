"""
Database layer — SQLAlchemy engine, session factory, and ORM models.
"""
from __future__ import annotations

import os
import socket
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()  # Load .env before any env-var reads


from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker


from src.utils.helpers import CONFIG, ROOT_DIR

# ── Connection URL ─────────────────────────────────────────────
# Priority: USE_SQLITE=1 env var → config.yaml url → env vars → SQLite fallback
_IS_VERCEL: bool = bool(os.getenv("VERCEL"))
_ENV_DB_URL: str = os.getenv("DATABASE_URL") or os.getenv("DB_URL") or ""
_USE_SQLITE: bool = os.getenv("USE_SQLITE", "0") == "1" or (_IS_VERCEL and not _ENV_DB_URL)

_PG_URL: str = CONFIG.get("database", {}).get(
    "url",
    "postgresql://{user}:{pw}@{host}:{port}/{db}".format(
        user=os.getenv("DB_USER", "pharma_user"),
        pw=os.getenv("DB_PASSWORD", "pharma_pass"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        db=os.getenv("DB_NAME", "pharmacovigilance"),
    ),
)
_SQLITE_PATH: Path = Path("/tmp/pharma_local.db") if _IS_VERCEL else (ROOT_DIR / "pharma_local.db")
_SQLITE_URL: str = "sqlite:///" + str(_SQLITE_PATH)

_DB_URL: str = _ENV_DB_URL or (_SQLITE_URL if _USE_SQLITE else _PG_URL)

# ── Engine ─────────────────────────────────────────────────────
_connect_args = {"check_same_thread": False} if _DB_URL.startswith("sqlite") else {}
if _IS_VERCEL and _DB_URL.startswith("postgresql"):
    try:
        parsed = urlparse(_DB_URL)
        host = parsed.hostname
        port = parsed.port or 5432
        if host:
            ipv4 = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]
            _connect_args["hostaddr"] = ipv4
    except Exception:
        # If resolution fails, keep default behavior and let DB errors surface in logs.
        pass

_engine_kwargs: dict = {"connect_args": _connect_args}
if not _DB_URL.startswith("sqlite"):
    _engine_kwargs["pool_pre_ping"] = True
    _engine_kwargs["pool_size"] = 10

engine = create_engine(_DB_URL, **_engine_kwargs)
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


class Submission(Base):
    """Regulatory submission tracking."""

    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String(128), index=True, nullable=False)
    submission_type = Column(String(32), nullable=False)  # PSUR / DSUR / PBRER / RMP
    status = Column(String(32), default="PENDING")  # PENDING / SUBMITTED / APPROVED / REJECTED
    due_date = Column(DateTime, nullable=False)
    submitted_date = Column(DateTime, nullable=True)
    complexity_weight = Column(Float, default=1.0)
    risk_score = Column(Float, default=0.0)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class CapaCase(Base):
    """Corrective and Preventive Action (CAPA) case."""

    __tablename__ = "capa_cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String(128), index=True, nullable=False)
    title = Column(String(512), nullable=False)
    description = Column(Text, nullable=True)
    state = Column(String(32), default="OPEN")  # OPEN / INVESTIGATION / CORRECTIVE_ACTION / VERIFICATION / CLOSED
    priority = Column(String(16), default="MEDIUM")  # CRITICAL / HIGH / MEDIUM / LOW
    assigned_to = Column(String(256), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    due_date = Column(DateTime, nullable=True)


class QCReport(Base):
    """Quality Control inspection report."""

    __tablename__ = "qc_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String(128), index=True, nullable=False)
    total_inspections = Column(Integer, default=0)
    defects_found = Column(Integer, default=0)
    inspector = Column(String(256), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class SystemEvent(Base):
    """Immutable audit trail entry for any operational action."""

    __tablename__ = "system_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    entity_type = Column(String(64), index=True, nullable=False)   # submission | capa | qc | audit
    entity_id = Column(String(64), index=True, nullable=False)     # PK of the related entity
    event_type = Column(String(64), nullable=False)                 # created | updated | status_change …
    event_description = Column(Text, nullable=False)
    user_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)


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
