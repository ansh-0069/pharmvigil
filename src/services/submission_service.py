"""
Regulatory Submission Scheduler — Module 1

Manages submission deadlines, overdue detection, risk scoring,
and daily scheduled updates via APScheduler.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

from loguru import logger
from sqlalchemy import and_
from sqlalchemy.orm import Session

from src.database.db import Submission, SessionLocal

# ── Constants ────────────────────────────────────────
VALID_SUBMISSION_TYPES = {"PSUR", "DSUR", "PBRER", "RMP"}
VALID_STATUSES = {"PENDING", "SUBMITTED", "APPROVED", "REJECTED"}

COMPLEXITY_DEFAULTS: dict[str, float] = {
    "PSUR": 1.2,
    "DSUR": 1.0,
    "PBRER": 1.5,
    "RMP": 1.3,
}


# ── Service Functions ────────────────────────────────
def create_submission(
    product_id: str,
    submission_type: str,
    due_date: datetime,
    complexity_weight: Optional[float] = None,
    notes: Optional[str] = None,
) -> Submission:
    """Create a new regulatory submission record."""
    submission_type = submission_type.upper()
    if submission_type not in VALID_SUBMISSION_TYPES:
        raise ValueError(
            f"Invalid submission type '{submission_type}'. "
            f"Must be one of {VALID_SUBMISSION_TYPES}"
        )

    if complexity_weight is None:
        complexity_weight = COMPLEXITY_DEFAULTS.get(submission_type, 1.0)

    submission = Submission(
        product_id=product_id,
        submission_type=submission_type,
        status="PENDING",
        due_date=due_date,
        complexity_weight=complexity_weight,
        notes=notes,
    )

    with SessionLocal() as session:
        session.add(submission)
        session.commit()
        session.refresh(submission)
        logger.info(
            f"Created submission {submission.id}: "
            f"{submission_type} for product {product_id}, due {due_date.date()}"
        )
        return submission


def update_submission_status(
    submission_id: int,
    new_status: str,
    submitted_date: Optional[datetime] = None,
) -> Submission:
    """Update the status of a submission."""
    new_status = new_status.upper()
    if new_status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{new_status}'. Must be one of {VALID_STATUSES}"
        )

    with SessionLocal() as session:
        sub = session.query(Submission).filter(Submission.id == submission_id).first()
        if not sub:
            raise ValueError(f"Submission {submission_id} not found")

        sub.status = new_status
        if new_status == "SUBMITTED":
            sub.submitted_date = submitted_date or datetime.utcnow()
        session.commit()
        session.refresh(sub)
        logger.info(f"Submission {submission_id} status → {new_status}")
        return sub


def get_overdue_submissions() -> List[dict]:
    """Return all submissions that are past due and not yet submitted."""
    now = datetime.utcnow()
    with SessionLocal() as session:
        rows = (
            session.query(Submission)
            .filter(
                and_(
                    Submission.due_date < now,
                    Submission.status == "PENDING",
                )
            )
            .order_by(Submission.due_date.asc())
            .all()
        )
        return [_submission_to_dict(r) for r in rows]


def calculate_submission_risk(submission_id: int) -> float:
    """
    Calculate risk score for a single submission.

    Formula: risk_score = complexity_weight * (1 / max(days_remaining, 1))
    """
    with SessionLocal() as session:
        sub = session.query(Submission).filter(Submission.id == submission_id).first()
        if not sub:
            raise ValueError(f"Submission {submission_id} not found")

        if sub.status in ("SUBMITTED", "APPROVED"):
            sub.risk_score = 0.0
        else:
            days_remaining = (sub.due_date - datetime.utcnow()).days
            sub.risk_score = round(
                sub.complexity_weight * (1.0 / max(days_remaining, 1)), 4
            )

        session.commit()
        session.refresh(sub)
        return sub.risk_score


def get_all_submissions() -> List[dict]:
    """Return all submissions."""
    with SessionLocal() as session:
        rows = session.query(Submission).order_by(Submission.due_date.asc()).all()
        return [_submission_to_dict(r) for r in rows]


# ── Scheduler Task ───────────────────────────────────
def update_all_risk_scores() -> None:
    """
    Daily scheduled task — recalculate risk scores for all PENDING submissions.
    Called by APScheduler.
    """
    logger.info("⏰ Running daily risk-score update …")
    now = datetime.utcnow()

    with SessionLocal() as session:
        pending = (
            session.query(Submission)
            .filter(Submission.status == "PENDING")
            .all()
        )
        for sub in pending:
            days_remaining = (sub.due_date - now).days
            sub.risk_score = round(
                sub.complexity_weight * (1.0 / max(days_remaining, 1)), 4
            )
        session.commit()
        logger.info(f"Updated risk scores for {len(pending)} pending submissions")


# ── Helpers ──────────────────────────────────────────
def _submission_to_dict(sub: Submission) -> dict:
    return {
        "id": sub.id,
        "product_id": sub.product_id,
        "submission_type": sub.submission_type,
        "status": sub.status,
        "due_date": sub.due_date.isoformat() if sub.due_date else None,
        "submitted_date": sub.submitted_date.isoformat() if sub.submitted_date else None,
        "complexity_weight": sub.complexity_weight,
        "risk_score": sub.risk_score,
        "notes": sub.notes,
        "created_at": sub.created_at.isoformat() if sub.created_at else None,
        "updated_at": sub.updated_at.isoformat() if sub.updated_at else None,
    }
