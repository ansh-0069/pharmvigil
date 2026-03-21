"""
Event Logging Service — Audit Trail Module

Provides a centralised, asynchronous-safe logging layer for all
operational actions in the PV Compliance platform.

Usage (from any service):
    from src.services.event_service import log_event

    log_event(
        entity_type="submission",
        entity_id="42",
        event_type="created",
        description="PSUR submission created for PROD-001 due 2025-06-30",
    )
"""
from __future__ import annotations

import html
import re
import uuid
from datetime import datetime
from typing import List, Optional

from loguru import logger

from src.database.db import SystemEvent, SessionLocal


def _sanitize_description(text: str) -> str:
    """Normalize event descriptions to readable plain text."""
    if not text:
        return ""
    cleaned = str(text)
    # Handle single or double-escaped payloads from legacy entries.
    for _ in range(3):
        unescaped = html.unescape(cleaned)
        if unescaped == cleaned:
            break
        cleaned = unescaped
    cleaned = re.sub(r"<br\s*/?>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    return " ".join(cleaned.split())


# ── Public API ───────────────────────────────────────
def log_event(
    entity_type: str,
    entity_id: str,
    event_type: str,
    description: str,
    user_id: Optional[str] = None,
) -> dict:
    """
    Persist a single audit trail event.

    Parameters
    ----------
    entity_type  : Category of the entity  (e.g. "submission", "capa", "qc")
    entity_id    : Identifier of the entity (int or UUID as string)
    event_type   : Short action label       (e.g. "created", "status_change")
    description  : Human-readable summary of what happened
    user_id      : Optional actor identifier

    Returns
    -------
    Dict representation of the persisted SystemEvent row.
    """
    description = _sanitize_description(description)
    event = SystemEvent(
        id=str(uuid.uuid4()),
        entity_type=entity_type.lower(),
        entity_id=str(entity_id),
        event_type=event_type.lower(),
        event_description=description,
        user_id=str(user_id) if user_id else None,
    )
    try:
        with SessionLocal() as session:
            session.add(event)
            session.commit()
            session.refresh(event)
            logger.debug(
                f"[EVENT] {entity_type}/{entity_id} → {event_type}: {description}"
            )
            return _event_to_dict(event)
    except Exception as exc:
        # Never let logging failures crash the calling service
        logger.error(f"Failed to persist event: {exc}")
        return {}


def get_entity_events(entity_type: str, entity_id: str) -> List[dict]:
    """
    Return all events for a specific entity, newest first.

    Parameters
    ----------
    entity_type : e.g. "submission", "capa", "qc"
    entity_id   : The entity's primary key (as string)
    """
    with SessionLocal() as session:
        rows = (
            session.query(SystemEvent)
            .filter(
                SystemEvent.entity_type == entity_type.lower(),
                SystemEvent.entity_id == str(entity_id),
            )
            .order_by(SystemEvent.created_at.desc())
            .all()
        )
        return [_event_to_dict(r) for r in rows]


def get_recent_events(limit: int = 50) -> List[dict]:
    """
    Return the most recent audit trail events across all entities.

    Parameters
    ----------
    limit : Maximum number of events to return (default 50, max 500)
    """
    limit = min(max(limit, 1), 500)
    with SessionLocal() as session:
        rows = (
            session.query(SystemEvent)
            .order_by(SystemEvent.created_at.desc())
            .limit(limit)
            .all()
        )
        return [_event_to_dict(r) for r in rows]


# ── Helpers ──────────────────────────────────────────
def _event_to_dict(event: SystemEvent) -> dict:
    return {
        "id": event.id,
        "entity_type": event.entity_type,
        "entity_id": event.entity_id,
        "event_type": event.event_type,
        "event_description": _sanitize_description(event.event_description),
        "user_id": event.user_id,
        "created_at": event.created_at.isoformat() if event.created_at else None,
    }
