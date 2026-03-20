"""
CAPA API Routes — Corrective and Preventive Action endpoints.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.services import capa_service

router = APIRouter(prefix="/capa", tags=["CAPA"])


# ── Schemas ──────────────────────────────────────────
class CreateCapaRequest(BaseModel):
    product_id: str = Field(..., min_length=1, examples=["PROD-001"])
    title: str = Field(..., min_length=1, examples=["Contamination in Batch #42"])
    description: Optional[str] = None
    assigned_to: Optional[str] = None
    due_date: Optional[datetime] = None
    priority: Optional[str] = Field(
        default="MEDIUM",
        description="CRITICAL / HIGH / MEDIUM / LOW",
        examples=["HIGH"],
    )


class UpdateCapaStatusRequest(BaseModel):
    state: str = Field(
        ...,
        examples=["INVESTIGATION"],
        description="Target state: INVESTIGATION, CORRECTIVE_ACTION, VERIFICATION, or CLOSED",
    )


class CapaCaseResponse(BaseModel):
    id: int
    product_id: str
    title: str
    description: Optional[str]
    state: str
    priority: Optional[str]
    assigned_to: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    due_date: Optional[str]
    next_state: Optional[str]


# ── CRUD ─────────────────────────────────────────────
@router.post("", response_model=CapaCaseResponse, status_code=201)
async def create_capa_case(body: CreateCapaRequest):
    """Create a new CAPA case."""
    return capa_service.create_capa_case(
        product_id=body.product_id,
        title=body.title,
        description=body.description,
        assigned_to=body.assigned_to,
        due_date=body.due_date,
        priority=body.priority or "MEDIUM",
    )


@router.get("/all", response_model=List[CapaCaseResponse])
async def get_all_capa_cases():
    """List ALL CAPA cases including CLOSED — used for the Kanban board."""
    return capa_service.get_all_capa_cases()


@router.get("/open", response_model=List[CapaCaseResponse])
async def get_open_capa_cases():
    """List all non-closed CAPA cases."""
    return capa_service.get_open_capa_cases()


@router.get("/overdue", response_model=List[CapaCaseResponse])
async def get_overdue_capa_cases():
    """List CAPA cases open longer than 30 days."""
    return capa_service.get_overdue_capa_cases()


# ── Generic status update (legacy) ───────────────────
@router.patch("/{case_id}/status", response_model=CapaCaseResponse)
async def update_capa_status(case_id: int, body: UpdateCapaStatusRequest):
    """Generic state transition endpoint."""
    try:
        return capa_service.update_capa_status(case_id, body.state)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Named semantic transition endpoints ──────────────
@router.patch("/{case_id}/investigate", response_model=CapaCaseResponse)
async def investigate_capa(case_id: int):
    """Advance CAPA from OPEN → INVESTIGATION."""
    try:
        return capa_service.transition_to_investigation(case_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.patch("/{case_id}/correct", response_model=CapaCaseResponse)
async def correct_capa(case_id: int):
    """Advance CAPA from INVESTIGATION → CORRECTIVE_ACTION."""
    try:
        return capa_service.transition_to_corrective_action(case_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.patch("/{case_id}/verify", response_model=CapaCaseResponse)
async def verify_capa(case_id: int):
    """Advance CAPA from CORRECTIVE_ACTION → VERIFICATION."""
    try:
        return capa_service.transition_to_verification(case_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.patch("/{case_id}/close", response_model=CapaCaseResponse)
async def close_capa(case_id: int):
    """Advance CAPA from VERIFICATION → CLOSED."""
    try:
        return capa_service.close_capa_case(case_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
