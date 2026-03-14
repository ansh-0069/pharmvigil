"""
Submission API Routes — regulatory submission endpoints.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.services import submission_service

router = APIRouter(prefix="/submissions", tags=["Submissions"])


# ── Schemas ──────────────────────────────────────────
class CreateSubmissionRequest(BaseModel):
    product_id: str = Field(..., min_length=1, examples=["PROD-001"])
    submission_type: str = Field(..., examples=["PSUR"])
    due_date: datetime = Field(..., examples=["2025-06-30T00:00:00"])
    complexity_weight: Optional[float] = Field(None, ge=0.1, le=10.0)
    notes: Optional[str] = None


class UpdateStatusRequest(BaseModel):
    status: str = Field(..., examples=["SUBMITTED"])
    submitted_date: Optional[datetime] = None


class SubmissionResponse(BaseModel):
    id: int
    product_id: str
    submission_type: str
    status: str
    due_date: Optional[str]
    submitted_date: Optional[str]
    complexity_weight: float
    risk_score: float
    notes: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


class RiskScoreResponse(BaseModel):
    submission_id: int
    risk_score: float


# ── Endpoints ────────────────────────────────────────
@router.post("", response_model=SubmissionResponse, status_code=201)
async def create_submission(body: CreateSubmissionRequest):
    """Create a new regulatory submission."""
    try:
        sub = submission_service.create_submission(
            product_id=body.product_id,
            submission_type=body.submission_type,
            due_date=body.due_date,
            complexity_weight=body.complexity_weight,
            notes=body.notes,
        )
        return submission_service._submission_to_dict(sub)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.patch("/{submission_id}/status", response_model=SubmissionResponse)
async def update_submission_status(submission_id: int, body: UpdateStatusRequest):
    """Update the status of a submission."""
    try:
        sub = submission_service.update_submission_status(
            submission_id=submission_id,
            new_status=body.status,
            submitted_date=body.submitted_date,
        )
        return submission_service._submission_to_dict(sub)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/overdue", response_model=List[SubmissionResponse])
async def get_overdue_submissions():
    """List all overdue submissions (past due, not yet submitted)."""
    return submission_service.get_overdue_submissions()


@router.get("", response_model=List[SubmissionResponse])
async def list_submissions():
    """List all submissions."""
    return submission_service.get_all_submissions()


@router.get("/{submission_id}/risk", response_model=RiskScoreResponse)
async def get_submission_risk(submission_id: int):
    """Calculate and return the risk score for a submission."""
    try:
        score = submission_service.calculate_submission_risk(submission_id)
        return RiskScoreResponse(submission_id=submission_id, risk_score=score)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
