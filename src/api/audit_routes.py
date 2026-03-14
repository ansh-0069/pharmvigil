"""
Audit Readiness API Routes — compliance metrics and audit scores.
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.services import audit_service

router = APIRouter(prefix="/audit", tags=["Audit"])


# ── Schemas ──────────────────────────────────────────
class AuditMetricsResponse(BaseModel):
    product_id: str
    on_time_submission_rate: float
    capa_closure_rate: float
    qc_defect_rate: float
    total_submissions: int
    total_capa_cases: int
    total_inspections: int
    audit_score: float


# ── Endpoints ────────────────────────────────────────
@router.get("/portfolio", response_model=List[AuditMetricsResponse])
async def get_portfolio_audit_scores():
    """Get audit readiness scores for all products (worst-performing first)."""
    return audit_service.get_portfolio_audit_scores()


@router.get("/{product_id}", response_model=AuditMetricsResponse)
async def get_audit_metrics(product_id: str):
    """Calculate audit metrics for a specific product."""
    result = audit_service.calculate_audit_metrics(product_id)
    if result["total_submissions"] == 0 and result["total_capa_cases"] == 0 and result["total_inspections"] == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for product '{product_id}'",
        )
    return result
