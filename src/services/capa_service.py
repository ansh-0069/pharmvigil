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
    {"trigger": "begin_investigation",    "source": "OPEN",             "dest": "INVESTIGATION"},
    {"trigger": "begin_corrective_action","source": "INVESTIGATION",    "dest": "CORRECTIVE_ACTION"},
    {"trigger": "begin_verification",     "source": "CORRECTIVE_ACTION","dest": "VERIFICATION"},
    {"trigger": "close_case",             "source": "VERIFICATION",     "dest": "CLOSED"},
]

# Map desired target state → trigger name
_STATE_TO_TRIGGER: dict[str, str] = {
    "INVESTIGATION":     "begin_investigation",
    "CORRECTIVE_ACTION": "begin_corrective_action",
    "VERIFICATION":      "begin_verification",
    "CLOSED":            "close_case",
}

# Valid next state for each current state
_NEXT_STATE: dict[str, str] = {
    "OPEN":             "INVESTIGATION",
    "INVESTIGATION":    "CORRECTIVE_ACTION",
    "CORRECTIVE_ACTION":"VERIFICATION",
    "VERIFICATION":     "CLOSED",
}

OVERDUE_DAYS = 30
VALID_PRIORITIES = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}


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


# ── Shared transition helper ─────────────────────────
def _do_transition(case_id: int, target_state: str) -> dict:
    """Core transition logic used by all named transition functions."""
    target_state = target_state.upper()
    with SessionLocal() as session:
        case = session.query(CapaCase).filter(CapaCase.id == case_id).first()
        if not case:
            raise ValueError(f"CAPA case {case_id} not found")

        sm = CapaStateMachine(initial_state=case.state)
        try:
            sm.transition_to(target_state)
        except Exception as exc:
            raise ValueError(
                f"Invalid transition: {case.state} → {target_state}. {exc}"
            ) from exc

        old_state = case.state
        case.state = target_state
        case.updated_at = datetime.utcnow()
        session.commit()
        session.refresh(case)

        logger.info(f"CAPA {case_id}: {old_state} → {target_state}")
        log_event(
            entity_type="capa",
            entity_id=str(case_id),
            event_type="status_change",
            description=(
                f"CAPA moved from {old_state} to {target_state} "
                f"(case: '{case.title}', product: '{case.product_id}')"
            ),
        )
        return _case_to_dict(case)


# ── Service Functions ────────────────────────────────
def create_capa_case(
    product_id: str,
    title: str,
    description: Optional[str] = None,
    assigned_to: Optional[str] = None,
    due_date: Optional[datetime] = None,
    priority: str = "MEDIUM",
) -> dict:
    """Create a new CAPA case in OPEN state."""
    if due_date is None:
        due_date = datetime.utcnow() + timedelta(days=OVERDUE_DAYS)
    priority = priority.upper() if priority else "MEDIUM"
    if priority not in VALID_PRIORITIES:
        priority = "MEDIUM"

    case = CapaCase(
        product_id=product_id,
        title=title,
        description=description,
        state="OPEN",
        priority=priority,
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
            description=(
                f"CAPA case opened: '{title}' for product '{product_id}' "
                f"[priority={priority}]"
            ),
        )
        return _case_to_dict(case)


def update_capa_status(case_id: int, new_state: str) -> dict:
    """Generic transition — validates via state machine."""
    return _do_transition(case_id, new_state)


# ── Named transition functions (explicit API surface) ─
def transition_to_investigation(capa_id: int) -> dict:
    """Move a CAPA case from OPEN → INVESTIGATION."""
    return _do_transition(capa_id, "INVESTIGATION")


def transition_to_corrective_action(capa_id: int) -> dict:
    """Move a CAPA case from INVESTIGATION → CORRECTIVE_ACTION."""
    return _do_transition(capa_id, "CORRECTIVE_ACTION")


def transition_to_verification(capa_id: int) -> dict:
    """Move a CAPA case from CORRECTIVE_ACTION → VERIFICATION."""
    return _do_transition(capa_id, "VERIFICATION")


def close_capa_case(capa_id: int) -> dict:
    """Move a CAPA case from VERIFICATION → CLOSED."""
    return _do_transition(capa_id, "CLOSED")


# ── Query Functions ───────────────────────────────────
def get_all_capa_cases() -> List[dict]:
    """Return ALL CAPA cases (including CLOSED) ordered by created_at desc."""
    with SessionLocal() as session:
        rows = (
            session.query(CapaCase)
            .order_by(CapaCase.created_at.desc())
            .all()
        )
        return [_case_to_dict(r) for r in rows]


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
        "id":          case.id,
        "product_id":  case.product_id,
        "title":       case.title,
        "description": case.description,
        "state":       case.state,
        "priority":    getattr(case, "priority", "MEDIUM") or "MEDIUM",
        "assigned_to": case.assigned_to,
        "created_at":  case.created_at.isoformat() if case.created_at else None,
        "updated_at":  case.updated_at.isoformat() if case.updated_at else None,
        "due_date":    case.due_date.isoformat() if case.due_date else None,
        "next_state":  _NEXT_STATE.get(case.state),
    }
