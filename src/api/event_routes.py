"""
Event API Routes — Audit trail endpoints.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from src.services import event_service

router = APIRouter(prefix="/api/v1/events", tags=["Audit Trail"])


# ── Schemas ──────────────────────────────────────────
class EventResponse(BaseModel):
    id: str
    entity_type: str
    entity_id: str
    event_type: str
    event_description: str
    user_id: Optional[str] = None
    created_at: Optional[str] = None


# ── Endpoints ────────────────────────────────────────
@router.get("/recent", response_model=List[EventResponse])
async def get_recent_events(
    limit: int = Query(default=50, ge=1, le=500, description="Max events to return"),
):
    """
    Return the most recent audit trail events across all entities.
    Suitable for an activity timeline in the Streamlit dashboard.
    """
    return event_service.get_recent_events(limit=limit)


@router.get("/{entity_type}/{entity_id}", response_model=List[EventResponse])
async def get_entity_events(entity_type: str, entity_id: str):
    """
    Return all audit events for a specific entity (e.g. a submission or CAPA case).

    entity_type: submission | capa | qc | audit
    entity_id:   the entity's primary key
    """
    return event_service.get_entity_events(entity_type=entity_type, entity_id=entity_id)
