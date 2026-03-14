"""
CAPA Workflow Engine — Module 2

State-machine-driven lifecycle management for Corrective and
Preventive Action (CAPA) cases using the `transitions` library.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

from loguru import logger
from sqlalchemy import and_
from transitions import Machine

from src.database.db import CapaCase, SessionLocal
from src.services.event_service import log_event

# ── State Machine Definition ─────────────────────────
CAPA_STATES = ["OPEN", "INVESTIGATION", "CORRECTIVE_ACTION", "VERIFICATION", "CLOSED"]

CAPA_TRANSITIONS = [
    {"trigger": "begin_investigation", "source": "OPEN", "dest": "INVESTIGATION"},
    {"trigger": "begin_corrective_action", "source": "INVESTIGATION", "dest": "CORRECTIVE_ACTION"},
    {"trigger": "begin_verification", "source": "CORRECTIVE_ACTION", "dest": "VERIFICATION"},
    {"trigger": "close_case", "source": "VERIFICATION", "dest": "CLOSED"},
]

# Map desired target state → trigger name
_STATE_TO_TRIGGER: dict[str, str] = {
    "INVESTIGATION": "begin_investigation",
    "CORRECTIVE_ACTION": "begin_corrective_action",
    "VERIFICATION": "begin_verification",
    "CLOSED": "close_case",
}

OVERDUE_DAYS = 30


class CapaStateMachine:
    """Lightweight wrapper around the transitions library for CAPA lifecycle."""

    def __init__(self, initial_state: str = "OPEN"):
        self.machine = Machine(
            model=self,
            states=CAPA_STATES,
            transitions=CAPA_TRANSITIONS,
            initial=initial_state,
            auto_transitions=False,
            send_event=False,
        )

    def transition_to(self, target_state: str) -> None:
        """Attempt to transition to the target state."""
        target_state = target_state.upper()
        trigger = _STATE_TO_TRIGGER.get(target_state)
        if trigger is None:
            raise ValueError(
                f"Cannot transition to '{target_state}'. "
                f"Valid targets: {list(_STATE_TO_TRIGGER.keys())}"
            )
        # The trigger raises MachineError if transition is invalid
        getattr(self, trigger)()


# ── Service Functions ────────────────────────────────
def create_capa_case(
    product_id: str,
    title: str,
    description: Optional[str] = None,
    assigned_to: Optional[str] = None,
    due_date: Optional[datetime] = None,
) -> dict:
    """Create a new CAPA case in OPEN state."""
    if due_date is None:
        due_date = datetime.utcnow() + timedelta(days=OVERDUE_DAYS)

    case = CapaCase(
        product_id=product_id,
        title=title,
        description=description,
        state="OPEN",
        assigned_to=assigned_to,
        due_date=due_date,
    )

    with SessionLocal() as session:
        session.add(case)
        session.commit()
        session.refresh(case)
        logger.info(f"Created CAPA case {case.id}: '{title}' for product {product_id}")
        log_event(
            entity_type="capa",
            entity_id=str(case.id),
            event_type="created",
            description=f"CAPA case opened: '{title}' for product '{product_id}'",
        )
        return _case_to_dict(case)


def update_capa_status(case_id: int, new_state: str) -> dict:
    """
    Transition a CAPA case to a new state.

    Uses the state machine to validate the transition is legal.
    """
    new_state = new_state.upper()

    with SessionLocal() as session:
        case = session.query(CapaCase).filter(CapaCase.id == case_id).first()
        if not case:
            raise ValueError(f"CAPA case {case_id} not found")

        # Validate transition via state machine
        sm = CapaStateMachine(initial_state=case.state)
        try:
            sm.transition_to(new_state)
        except Exception as exc:
            raise ValueError(
                f"Invalid transition: {case.state} → {new_state}. {exc}"
            ) from exc

        old_state = case.state
        case.state = new_state
        case.updated_at = datetime.utcnow()
        session.commit()
        session.refresh(case)
        logger.info(f"CAPA {case_id}: {old_state} → {new_state}")
        log_event(
            entity_type="capa",
            entity_id=str(case_id),
            event_type="status_change",
            description=(
                f"CAPA status changed: {old_state} → {new_state} "
                f"(case: '{case.title}', product: '{case.product_id}')"
            ),
        )
        return _case_to_dict(case)


def get_open_capa_cases() -> List[dict]:
    """Return all CAPA cases that are not CLOSED."""
    with SessionLocal() as session:
        rows = (
            session.query(CapaCase)
            .filter(CapaCase.state != "CLOSED")
            .order_by(CapaCase.created_at.desc())
            .all()
        )
        return [_case_to_dict(r) for r in rows]


def get_overdue_capa_cases() -> List[dict]:
    """Return CAPA cases open longer than 30 days."""
    cutoff = datetime.utcnow() - timedelta(days=OVERDUE_DAYS)
    with SessionLocal() as session:
        rows = (
            session.query(CapaCase)
            .filter(
                and_(
                    CapaCase.state != "CLOSED",
                    CapaCase.created_at < cutoff,
                )
            )
            .order_by(CapaCase.created_at.asc())
            .all()
        )
        return [_case_to_dict(r) for r in rows]


# ── Helpers ──────────────────────────────────────────
def _case_to_dict(case: CapaCase) -> dict:
    return {
        "id": case.id,
        "product_id": case.product_id,
        "title": case.title,
        "description": case.description,
        "state": case.state,
        "assigned_to": case.assigned_to,
        "created_at": case.created_at.isoformat() if case.created_at else None,
        "updated_at": case.updated_at.isoformat() if case.updated_at else None,
        "due_date": case.due_date.isoformat() if case.due_date else None,
    }
