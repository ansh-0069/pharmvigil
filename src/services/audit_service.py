"""
Audit Readiness Engine — Module 3

Calculates compliance metrics and audit readiness scores
for individual products and the entire portfolio.
"""
from __future__ import annotations

from typing import List

from loguru import logger
from sqlalchemy import func

from src.database.db import CapaCase, QCReport, Submission, SessionLocal
from src.services.event_service import log_event


# ── Service Functions ────────────────────────────────
def calculate_audit_metrics(product_id: str) -> dict:
    """
    Calculate compliance metrics for a single product.

    Metrics:
        on_time_submission_rate — fraction of submissions submitted on time
        capa_closure_rate       — fraction of CAPA cases that are CLOSED
        qc_defect_rate          — defects / total inspections

    Audit readiness score:
        0.5 * on_time_rate + 0.3 * capa_closure_rate + 0.2 * (1 - qc_defect_rate)
    """
    with SessionLocal() as session:
        # ── Submission metrics ───────────────────
        total_subs = (
            session.query(func.count(Submission.id))
            .filter(Submission.product_id == product_id)
            .scalar()
        ) or 0

        on_time_subs = (
            session.query(func.count(Submission.id))
            .filter(
                Submission.product_id == product_id,
                Submission.status.in_(["SUBMITTED", "APPROVED"]),
                Submission.submitted_date <= Submission.due_date,
            )
            .scalar()
        ) or 0

        on_time_rate = on_time_subs / max(total_subs, 1)

        # ── CAPA metrics ────────────────────────
        total_capas = (
            session.query(func.count(CapaCase.id))
            .filter(CapaCase.product_id == product_id)
            .scalar()
        ) or 0

        closed_capas = (
            session.query(func.count(CapaCase.id))
            .filter(
                CapaCase.product_id == product_id,
                CapaCase.state == "CLOSED",
            )
            .scalar()
        ) or 0

        capa_closure_rate = closed_capas / max(total_capas, 1)

        # ── QC metrics ──────────────────────────
        qc_agg = (
            session.query(
                func.coalesce(func.sum(QCReport.total_inspections), 0),
                func.coalesce(func.sum(QCReport.defects_found), 0),
            )
            .filter(QCReport.product_id == product_id)
            .one()
        )
        total_inspections, total_defects = int(qc_agg[0]), int(qc_agg[1])
        qc_defect_rate = total_defects / max(total_inspections, 1)

        # ── Audit readiness score ────────────────
        audit_score = round(
            0.5 * on_time_rate
            + 0.3 * capa_closure_rate
            + 0.2 * (1.0 - qc_defect_rate),
            4,
        )

        result = {
            "product_id": product_id,
            "on_time_submission_rate": round(on_time_rate, 4),
            "capa_closure_rate": round(capa_closure_rate, 4),
            "qc_defect_rate": round(qc_defect_rate, 4),
            "total_submissions": total_subs,
            "total_capa_cases": total_capas,
            "total_inspections": total_inspections,
            "audit_score": audit_score,
        }
        logger.info(f"Audit metrics for {product_id}: score={audit_score}")
        log_event(
            entity_type="audit",
            entity_id=product_id,
            event_type="audit_score_recalculated",
            description=(
                f"Audit score recalculated for '{product_id}': {audit_score} "
                f"(on_time={on_time_rate:.2%}, capa_closure={capa_closure_rate:.2%}, "
                f"qc_defect={qc_defect_rate:.2%})"
            ),
        )
        return result


def get_portfolio_audit_scores() -> List[dict]:
    """
    Calculate audit readiness scores across all products.

    Returns a list of per-product audit metric dicts, sorted by
    audit_score ascending (worst-performing first).
    """
    product_ids = _get_all_product_ids()
    if not product_ids:
        return []

    results = [calculate_audit_metrics(pid) for pid in product_ids]
    results.sort(key=lambda r: r["audit_score"])
    return results


# ── Helpers ──────────────────────────────────────────
def _get_all_product_ids() -> List[str]:
    """Collect distinct product IDs across submissions, CAPA, and QC tables."""
    with SessionLocal() as session:
        ids: set[str] = set()

        for model in (Submission, CapaCase, QCReport):
            rows = session.query(model.product_id).distinct().all()
            ids.update(r[0] for r in rows)

        return sorted(ids)
