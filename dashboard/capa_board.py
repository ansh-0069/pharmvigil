"""
CAPA Kanban Workflow Board
Enterprise-grade visual pipeline for CAPA lifecycle management.
"""
from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────
API_URL = "http://localhost:8000"

STATES = ["OPEN", "INVESTIGATION", "CORRECTIVE_ACTION", "VERIFICATION", "CLOSED"]

# Endpoint called to advance to each state
_ADVANCE_ENDPOINT = {
    "INVESTIGATION":     "investigate",
    "CORRECTIVE_ACTION": "correct",
    "VERIFICATION":      "verify",
    "CLOSED":            "close",
}

_STATE_LABEL = {
    "OPEN":             "📂 OPEN",
    "INVESTIGATION":    "🔎 INVESTIGATION",
    "CORRECTIVE_ACTION":"🔧 CORRECTIVE ACTION",
    "VERIFICATION":     "✅ VERIFICATION",
    "CLOSED":           "🔒 CLOSED",
}

_STATE_COLOR = {
    "OPEN":              "#3877f6",
    "INVESTIGATION":     "#f5a623",
    "CORRECTIVE_ACTION": "#a855f7",
    "VERIFICATION":      "#06b6d4",
    "CLOSED":            "#22c55e",
}

_PRIORITY_COLOR = {
    "CRITICAL": "#ef4444",
    "HIGH":     "#f5a623",
    "MEDIUM":   "#3877f6",
    "LOW":      "#22c55e",
}

_PRIORITY_ICON = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🔵",
    "LOW":      "🟢",
}


# ── CSS ───────────────────────────────────────────────
BOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono&display=swap');

/* Board global */
.kb-board { display:flex; gap:0; }

/* Column header */
.kb-col-header {
    border-radius:10px 10px 0 0;
    padding:10px 14px;
    font-family:'Inter',sans-serif;
    font-size:0.78rem;
    font-weight:700;
    letter-spacing:0.06em;
    text-transform:uppercase;
    margin-bottom:10px;
    display:flex;
    align-items:center;
    justify-content:space-between;
}

/* Card */
.kb-card {
    background:#0d1320;
    border:1px solid rgba(255,255,255,0.07);
    border-radius:10px;
    padding:14px 14px 12px;
    margin-bottom:10px;
    transition:transform 0.15s, box-shadow 0.15s;
    position:relative;
}
.kb-card:hover {
    transform:translateY(-2px);
    box-shadow:0 8px 24px rgba(0,0,0,0.35);
}
.kb-card-id {
    font-family:'JetBrains Mono',monospace;
    font-size:0.68rem;
    color:#55657d;
    margin-bottom:5px;
}
.kb-card-title {
    font-family:'Inter',sans-serif;
    font-size:0.88rem;
    font-weight:600;
    color:#d0d8e8;
    margin-bottom:8px;
    line-height:1.35;
}
.kb-card-meta {
    font-size:0.72rem;
    color:#6b7fa0;
    margin-bottom:6px;
}
.kb-badge {
    display:inline-block;
    font-size:0.65rem;
    font-weight:700;
    letter-spacing:0.05em;
    padding:2px 8px;
    border-radius:20px;
    margin-bottom:6px;
}

/* Empty column placeholder */
.kb-empty {
    border:1px dashed rgba(255,255,255,0.08);
    border-radius:10px;
    padding:24px 10px;
    text-align:center;
    color:#3a4a66;
    font-size:0.78rem;
    font-family:'Inter',sans-serif;
}

/* Column count badge */
.kb-count {
    background:rgba(255,255,255,0.08);
    border-radius:20px;
    padding:2px 8px;
    font-size:0.72rem;
    font-family:'JetBrains Mono',monospace;
}
</style>
"""


# ── Data Fetchers ─────────────────────────────────────
@st.cache_data(ttl=15)
def _fetch_all_cases() -> list:
    try:
        r = requests.get(f"{API_URL}/capa/all", timeout=5)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def _advance_case(case_id: int, next_state: str) -> bool:
    """POST to the semantic transition endpoint."""
    endpoint = _ADVANCE_ENDPOINT.get(next_state)
    if not endpoint:
        return False
    try:
        r = requests.patch(f"{API_URL}/capa/{case_id}/{endpoint}", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _create_case(product_id: str, title: str, description: str,
                 priority: str, assigned_to: str) -> bool:
    try:
        r = requests.post(
            f"{API_URL}/capa",
            json={
                "product_id": product_id,
                "title": title,
                "description": description,
                "priority": priority,
                "assigned_to": assigned_to or None,
            },
            timeout=5,
        )
        return r.status_code == 201
    except Exception:
        return False


# ── Card renderer ─────────────────────────────────────
def _render_card(case: dict, col_state: str) -> None:
    cid = case.get("id", "?")
    title = case.get("title", "Untitled")[:80]
    desc = (case.get("description") or "")[:100]
    prio = (case.get("priority") or "MEDIUM").upper()
    assigned = case.get("assigned_to") or "Unassigned"
    due = case.get("due_date", "")
    due_str = pd.to_datetime(due).strftime("%b %d, %Y") if due else "—"
    next_state = case.get("next_state")

    prio_color = _PRIORITY_COLOR.get(prio, "#3877f6")
    prio_icon  = _PRIORITY_ICON.get(prio, "🔵")

    # Overdue check
    is_overdue = False
    if due and col_state != "CLOSED":
        try:
            is_overdue = pd.to_datetime(due) < pd.Timestamp.utcnow()
        except Exception:
            pass

    border_color = "#ef4444" if is_overdue else "rgba(255,255,255,0.07)"

    st.markdown(f"""
    <div class="kb-card" style="border-color:{border_color}">
        <div class="kb-card-id">CAPA-{cid:03d}{" · ⚠️ OVERDUE" if is_overdue else ""}</div>
        <div class="kb-card-title">{title}</div>
        {f'<div class="kb-card-meta" style="font-style:italic;color:#4a596e;">{desc}</div>' if desc else ""}
        <span class="kb-badge" style="background:{prio_color}22;color:{prio_color};border:1px solid {prio_color}44;">
            {prio_icon} {prio}
        </span>
        <div class="kb-card-meta">👤 {assigned}</div>
        <div class="kb-card-meta">📅 Due: {due_str}</div>
    </div>
    """, unsafe_allow_html=True)

    # Advance button (only if there's a valid next state)
    if next_state and col_state != "CLOSED":
        btn_label = f"→ {_STATE_LABEL.get(next_state, next_state)}"
        btn_key = f"advance_{cid}_{col_state}_{int(time.time() * 1000) % 100000}"
        if st.button(btn_label, key=btn_key, use_container_width=True):
            with st.spinner("Transitioning …"):
                ok = _advance_case(cid, next_state)
            if ok:
                st.success(f"CAPA-{cid:03d} moved to {next_state}")
                _fetch_all_cases.clear()
                st.rerun()
            else:
                st.error("Transition failed — check server logs.")


# ── Main Board ────────────────────────────────────────
def render_capa_board() -> None:
    st.markdown(BOARD_CSS, unsafe_allow_html=True)

    # ── Header bar ───────────────────────────────────
    hd, btn_new, btn_refresh = st.columns([5, 1.5, 1])
    with hd:
        st.markdown(
            "<h2 style='font-family:Inter,sans-serif;margin-bottom:4px;'>"
            "⚡ CAPA Workflow Board</h2>"
            "<p style='color:#6b7fa0;font-size:0.82rem;margin-top:0;'>"
            "Drag-free Kanban pipeline — click a card's button to advance it to the next stage.</p>",
            unsafe_allow_html=True,
        )
    with btn_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            _fetch_all_cases.clear()
            st.rerun()
    with btn_new:
        if st.button("➕ New CAPA", use_container_width=True, type="primary"):
            st.session_state["show_new_capa_form"] = not st.session_state.get("show_new_capa_form", False)

    # ── New CAPA creation form ────────────────────────
    if st.session_state.get("show_new_capa_form", False):
        with st.expander("📝 Create New CAPA Case", expanded=True):
            fc1, fc2 = st.columns(2)
            with fc1:
                n_product  = st.text_input("Product ID *", placeholder="PROD-001")
                n_title    = st.text_input("Title *",       placeholder="e.g. Contamination in Batch #42")
                n_desc     = st.text_area("Description",    placeholder="Detailed issue description …", height=80)
            with fc2:
                n_priority = st.selectbox("Priority", ["CRITICAL","HIGH","MEDIUM","LOW"], index=2)
                n_assigned = st.text_input("Assigned To", placeholder="e.g. Dr. Smith")
            if st.button("💾 Create Case", type="primary"):
                if n_product and n_title:
                    ok = _create_case(n_product, n_title, n_desc, n_priority, n_assigned)
                    if ok:
                        st.success("CAPA case created!")
                        st.session_state["show_new_capa_form"] = False
                        _fetch_all_cases.clear()
                        time.sleep(0.4)
                        st.rerun()
                    else:
                        st.error("Failed to create case — ensure FastAPI is running.")
                else:
                    st.warning("Product ID and Title are required.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Fetch & bucket by state ───────────────────────
    all_cases = _fetch_all_cases()
    board: dict[str, list] = {s: [] for s in STATES}
    for c in all_cases:
        state = c.get("state", "OPEN")
        if state in board:
            board[state].append(c)

    # ── Kanban columns ────────────────────────────────
    cols = st.columns(5, gap="small")
    for col_widget, state in zip(cols, STATES):
        color = _STATE_COLOR[state]
        label = _STATE_LABEL[state]
        count = len(board[state])

        with col_widget:
            st.markdown(
                f'<div class="kb-col-header" style="background:{color}22;border:1px solid {color}44;color:{color};">'
                f'{label}<span class="kb-count" style="color:{color};">{count}</span></div>',
                unsafe_allow_html=True,
            )
            if not board[state]:
                st.markdown('<div class="kb-empty">No cases</div>', unsafe_allow_html=True)
            else:
                for case in board[state]:
                    _render_card(case, state)


# ── Standalone run support ────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="CAPA Workflow Board",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_capa_board()
