"""
AI Pharmacovigilance Intelligence Platform — Enterprise Dashboard v2.0

Professional-grade analytics dashboard for pharmaceutical safety monitoring.
Features: signal detection, drug risk heatmaps, network graph, geographic
risk map, adverse event trends, real-time prediction, system monitoring,
and CSV export capabilities.
"""
from __future__ import annotations

import io
import json
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

# ═══════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="PharmVigil AI — Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"
DATA_DIR = Path("data")
SCORES_PATH = DATA_DIR / "processed" / "signal_scores.csv"
RAW_PATH = DATA_DIR / "raw" / "adverse_events.csv"

# ── Color system ─────────────────────────────────────
C = {
    "bg":          "#060a13",
    "bg_card":     "#0d1320",
    "bg_elevated": "#111827",
    "bg_hover":    "#172033",
    "border":      "#1c2740",
    "border_glow": "rgba(56,119,246,0.18)",
    "text":        "#e8ecf4",
    "text_dim":    "#8896b0",
    "text_muted":  "#55657d",
    "blue":        "#3877f6",
    "blue_l":      "#6da0ff",
    "cyan":        "#00d4e5",
    "emerald":     "#10b981",
    "amber":       "#f5a623",
    "rose":        "#f43f5e",
    "violet":      "#a78bfa",
    "critical":    "#ef4444",
    "high":        "#f97316",
    "medium":      "#eab308",
    "low":         "#22c55e",
}

ALERT_MAP = {"critical": C["critical"], "high": C["high"], "medium": C["medium"], "low": C["low"]}


# ═══════════════════════════════════════════════════════
#  CSS INJECTION — Enterprise Dark Theme
# ═══════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Reset ─────────────────────────────── */
    *, html, body, [class*="st-"] { font-family: 'Inter', system-ui, sans-serif !important; }
    .main { background: #060a13; }
    .block-container { padding: 1rem 2rem 2rem !important; max-width: 1440px; }
    hr { border-color: rgba(56,119,246,0.08) !important; margin: 20px 0 !important; }

    /* ── Sidebar ───────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070c18 0%, #0d1525 100%);
        border-right: 1px solid rgba(56,119,246,0.08);
    }
    section[data-testid="stSidebar"] .stMarkdown { color: #8896b0; }

    /* ── Headings ──────────────────────────── */
    h1 { color: #e8ecf4 !important; font-weight: 800 !important; font-size: 1.55rem !important; letter-spacing: -0.03em; }
    h2 { color: #d0d8e8 !important; font-weight: 700 !important; font-size: 1.15rem !important; }
    h3 { color: #b0bcd0 !important; font-weight: 600 !important; font-size: 1rem !important; }

    /* ── Hero ───────────────────────────────── */
    .hero {
        background: linear-gradient(135deg, #0c1424 0%, #142042 40%, #0f1832 100%);
        border: 1px solid rgba(56,119,246,0.12);
        border-radius: 14px; padding: 22px 30px; margin-bottom: 20px;
        position: relative; overflow: hidden;
    }
    .hero::before {
        content: ''; position: absolute; top: -40px; right: -20px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(56,119,246,0.06) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 1.35rem; font-weight: 800; color: #e8ecf4;
        letter-spacing: -0.03em; margin: 0 0 4px 0;
    }
    .hero-sub {
        font-size: 0.82rem; color: #6b7fa0; line-height: 1.5;
        margin: 0 0 8px 0; max-width: 700px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(56,119,246,0.1); color: #6da0ff;
        padding: 3px 12px; border-radius: 14px;
        font-size: 0.67rem; font-weight: 700; letter-spacing: 0.06em;
        border: 1px solid rgba(56,119,246,0.2);
        text-transform: uppercase;
    }

    /* ── KPI Card ──────────────────────────── */
    .kpi {
        background: linear-gradient(135deg, #0d1320 0%, #111d32 100%);
        border: 1px solid rgba(56,119,246,0.08); border-radius: 12px;
        padding: 18px 20px; text-align: center;
        transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
        position: relative; overflow: hidden;
    }
    .kpi::after {
        content: ''; position: absolute; top: 0; left: 0; right: 0;
        height: 2px; border-radius: 12px 12px 0 0;
    }
    .kpi:hover {
        border-color: rgba(56,119,246,0.25);
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(56,119,246,0.06);
    }
    .kpi-icon { font-size: 22px; margin-bottom: 6px; opacity: 0.9; }
    .kpi-val {
        font-size: 1.9rem; font-weight: 900; letter-spacing: -0.04em;
        line-height: 1; margin-bottom: 3px;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .kpi-lbl {
        font-size: 0.68rem; color: #6b7fa0; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.1em;
    }
    .kpi-blue::after  { background: linear-gradient(90deg, #3877f6, #6da0ff); }
    .kpi-blue .kpi-val { color: #6da0ff; }
    .kpi-green::after  { background: linear-gradient(90deg, #10b981, #34d399); }
    .kpi-green .kpi-val { color: #34d399; }
    .kpi-amber::after  { background: linear-gradient(90deg, #f5a623, #fbbf24); }
    .kpi-amber .kpi-val { color: #fbbf24; }
    .kpi-rose::after   { background: linear-gradient(90deg, #f43f5e, #fb7185); }
    .kpi-rose .kpi-val  { color: #fb7185; }

    /* ── Panel card ────────────────────────── */
    .panel {
        background: #0d1320; border: 1px solid rgba(56,119,246,0.08);
        border-radius: 12px; padding: 20px 22px; margin-bottom: 12px;
    }
    .panel h4 {
        font-size: 0.85rem !important; color: #8896b0 !important;
        font-weight: 600 !important; margin: 0 0 12px 0 !important;
        text-transform: uppercase; letter-spacing: 0.06em;
    }

    /* ── Severity rows ─────────────────────── */
    .sev-row {
        display: flex; align-items: center; gap: 10px;
        padding: 8px 12px; border-radius: 8px; margin-bottom: 4px;
        background: rgba(17,24,39,0.5);
        border-left: 3px solid transparent;
        transition: background 0.2s;
    }
    .sev-row:hover { background: rgba(56,119,246,0.04); }
    .sev-critical { border-left-color: #ef4444; }
    .sev-high     { border-left-color: #f97316; }
    .sev-medium   { border-left-color: #eab308; }
    .sev-low      { border-left-color: #22c55e; }
    .sev-label { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; min-width: 65px; }
    .sev-drug  { font-size: 0.82rem; color: #d0d8e8; flex: 1; }
    .sev-score { font-size: 0.82rem; font-family: 'JetBrains Mono', monospace !important; }

    /* ── Status dot ────────────────────────── */
    .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
    .dot-green { background: #22c55e; box-shadow: 0 0 6px rgba(34,197,94,0.4); }
    .dot-amber { background: #f5a623; box-shadow: 0 0 6px rgba(245,166,35,0.4); }
    .dot-red   { background: #ef4444; box-shadow: 0 0 6px rgba(239,68,68,0.4); }

    /* ── Pred card ─────────────────────────── */
    .pred-card {
        background: linear-gradient(135deg, #0d1320 0%, #142042 100%);
        border: 1px solid rgba(56,119,246,0.12); border-radius: 14px;
        padding: 24px; text-align: center;
    }
    .pred-val {
        font-size: 2.8rem; font-weight: 900; letter-spacing: -0.04em;
        font-family: 'JetBrains Mono', monospace !important; line-height: 1;
    }
    .pred-lbl {
        font-size: 0.72rem; color: #6b7fa0; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.1em; margin-top: 5px;
    }

    /* ── Tabs ──────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px; background: rgba(13,19,32,0.6);
        border-radius: 10px; padding: 3px; border: 1px solid rgba(56,119,246,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px; padding: 8px 16px; color: #6b7fa0; font-weight: 500; font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56,119,246,0.1) !important; color: #6da0ff !important;
    }

    /* ── Scrollbar ─────────────────────────── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #060a13; }
    ::-webkit-scrollbar-thumb { background: #1c2740; border-radius: 4px; }

    /* ── Download btn ──────────────────────── */
    .stDownloadButton > button {
        background: rgba(56,119,246,0.1) !important; color: #6da0ff !important;
        border: 1px solid rgba(56,119,246,0.2) !important; border-radius: 8px !important;
        font-weight: 600 !important; font-size: 0.8rem !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(56,119,246,0.2) !important;
    }

    /* ── Inputs ────────────────────────────── */
    .stSelectbox > div > div, .stMultiSelect > div > div, .stTextInput > div > div {
        background: #0d1320 !important; border-color: rgba(56,119,246,0.12) !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  PLOTLY THEME HELPER
# ═══════════════════════════════════════════════════════
_PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,19,32,0.4)",
    font=dict(family="Inter, system-ui, sans-serif", color=C["text"], size=12),
    margin=dict(l=16, r=16, t=44, b=16),
    xaxis=dict(gridcolor="rgba(28,39,64,0.4)", zerolinecolor="rgba(28,39,64,0.4)"),
    yaxis=dict(gridcolor="rgba(28,39,64,0.4)", zerolinecolor="rgba(28,39,64,0.4)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11)),
    hoverlabel=dict(bgcolor=C["bg_elevated"], font_size=12, bordercolor=C["border"]),
)

def theme(fig: go.Figure, h: int = 420) -> go.Figure:
    fig.update_layout(**_PLOTLY_BASE, height=h)
    return fig


# ═══════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_scores() -> pd.DataFrame:
    if SCORES_PATH.exists():
        return pd.read_csv(SCORES_PATH)
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_raw() -> pd.DataFrame:
    if RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH)
        if "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
        return df
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════
#  COMPONENT HELPERS
# ═══════════════════════════════════════════════════════
def kpi(icon: str, val: str, label: str, variant: str = "blue"):
    st.markdown(f"""
    <div class="kpi kpi-{variant}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-val">{val}</div>
        <div class="kpi-lbl">{label}</div>
    </div>""", unsafe_allow_html=True)


def sev_row(level: str, drug: str, event: str, score: float):
    color = ALERT_MAP.get(level, C["blue"])
    st.markdown(f"""
    <div class="sev-row sev-{level}">
        <span class="sev-label" style="color:{color}">{level}</span>
        <span class="sev-drug"><b>{drug.title()}</b> → {event.title()}</span>
        <span class="sev-score" style="color:{color}">{score:.4f}</span>
    </div>""", unsafe_allow_html=True)


def api_predict(drug: str, event: str) -> tuple[Optional[dict], float]:
    """Returns (result_dict, response_time_ms)."""
    t0 = time.time()
    try:
        r = requests.post(f"{API_BASE}/predict",
                          json={"drug_name": drug, "adverse_event": event}, timeout=30)
        elapsed = (time.time() - t0) * 1000
        if r.status_code == 200:
            return r.json(), elapsed
    except (requests.ConnectionError, requests.Timeout, requests.exceptions.RequestException):
        elapsed = (time.time() - t0) * 1000
    return None, elapsed


def api_health() -> tuple[bool, float]:
    """Ping /health and return (is_healthy, ms)."""
    t0 = time.time()
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        ms = (time.time() - t0) * 1000
        return r.status_code == 200, ms
    except Exception:
        return False, (time.time() - t0) * 1000


def build_network_html(scores_df: pd.DataFrame, top_n: int = 40) -> str:
    """Build a drug-event network graph and return HTML string."""
    top = scores_df.nlargest(top_n, "risk_score")
    G = nx.Graph()

    for _, row in top.iterrows():
        drug = row["drug_name"].title()
        event = row["adverse_event"].title()
        score = row["risk_score"]
        alert = row.get("alert_level", "low")
        color = ALERT_MAP.get(alert, C["blue"])

        if drug not in G:
            G.add_node(drug, size=20, color="#6da0ff", title=f"Drug: {drug}",
                       font={"color": "#e8ecf4", "size": 12})
        if event not in G:
            G.add_node(event, size=14, color="#f5a623", title=f"Event: {event}",
                       font={"color": "#e8ecf4", "size": 10})

        G.add_edge(drug, event, value=score * 3, color=color,
                   title=f"Risk: {score:.4f} ({alert})")

    # Use pyvis
    from pyvis.network import Network
    net = Network(height="480px", width="100%", bgcolor="#0d1320",
                  font_color="#e8ecf4", directed=False)
    net.from_nx(G)
    net.set_options("""
    {
      "nodes": {"borderWidth": 0, "shadow": true},
      "edges": {"smooth": {"type": "continuous"}, "shadow": true},
      "physics": {
        "forceAtlas2Based": {"gravitationalConstant": -60, "springLength": 120},
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 80}
      },
      "interaction": {"hover": true, "tooltipDelay": 100}
    }
    """)
    # Write to temp file and read HTML
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8")
    net.save_graph(tmp.name)
    tmp.close()
    with open(tmp.name, "r", encoding="utf-8") as f:
        return f.read()


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
def main():
    inject_css()

    scores = load_scores()
    raw = load_raw()
    has_s = not scores.empty
    has_r = not raw.empty

    # ── HERO ─────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-title">🛡️ AI Pharmacovigilance Intelligence Platform</div>
        <p class="hero-sub">
            Enterprise drug safety signal detection — analysing FDA FAERS adverse
            event data with ML & NLP to surface emerging pharmacovigilance signals
            in real time.
        </p>
        <span class="hero-badge">⚡ Signal Engine v2.0</span>
    </div>""", unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        kpi("📋", f"{len(raw):,}" if has_r else "—", "Total Reports", "blue")
    with k2:
        kpi("💊", f"{raw['drug_name'].nunique():,}" if has_r and 'drug_name' in raw.columns else "—",
            "Unique Drugs", "green")
    with k3:
        kpi("⚠️", f"{raw['adverse_event'].nunique():,}" if has_r and 'adverse_event' in raw.columns else "—",
            "Adverse Events", "amber")
    with k4:
        hi = len(scores[scores["alert_level"].isin(["critical","high"])]) if has_s and "alert_level" in scores.columns else 0
        kpi("🚨", f"{hi:,}", "High Risk Signals", "rose")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── SIDEBAR ──────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:14px 0 4px">
            <span style="font-size:1.8rem">🛡️</span>
            <div style="font-size:1.05rem;font-weight:800;color:#e8ecf4;margin-top:2px">PharmVigil AI</div>
            <div style="font-size:0.65rem;color:#55657d;font-weight:500;letter-spacing:0.08em;text-transform:uppercase">Control Panel</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")

        drug_opts = sorted(raw["drug_name"].dropna().unique().tolist()) if has_r else []
        sel_drugs = st.multiselect("🔬 Filter by Drug", drug_opts, default=[], key="sb_drugs")

        event_opts = sorted(raw["adverse_event"].dropna().unique().tolist()) if has_r else []
        sel_events = st.multiselect("⚠️ Filter by Event", event_opts, default=[], key="sb_events")

        st.markdown("---")
        risk_thresh = st.slider("🎯 Risk Score Threshold", 0.0, 1.0, 0.6, 0.05, key="sb_thresh")

        if has_r and "report_date" in raw.columns:
            mn, mx = raw["report_date"].min(), raw["report_date"].max()
            if pd.notna(mn) and pd.notna(mx):
                date_range = st.date_input("📅 Date Range", value=(mn.date(), mx.date()),
                                           min_value=mn.date(), max_value=mx.date(), key="sb_date")
            else:
                date_range = None
        else:
            date_range = None

        st.markdown("---")

        # ── System monitoring panel ──────────────
        st.markdown('<div class="panel"><h4>🖥 System Monitor</h4>', unsafe_allow_html=True)
        api_ok, api_ms = api_health()
        dot = "dot-green" if api_ok else "dot-red"
        label = "Operational" if api_ok else "Offline"
        st.markdown(f'<span class="dot {dot}"></span><span style="color:#d0d8e8;font-size:0.82rem">API: {label}</span>'
                    f'<span style="float:right;color:#6b7fa0;font-size:0.75rem;font-family:JetBrains Mono,monospace">{api_ms:.0f}ms</span>',
                    unsafe_allow_html=True)

        model_exists = Path("models/xgboost_classifier.joblib").exists()
        mdot = "dot-green" if model_exists else "dot-amber"
        mlbl = "Loaded" if model_exists else "Not trained"
        st.markdown(f'<span class="dot {mdot}"></span><span style="color:#d0d8e8;font-size:0.82rem">Model: {mlbl}</span>',
                    unsafe_allow_html=True)

        if has_r and "report_date" in raw.columns:
            latest = raw["report_date"].max()
            if pd.notna(latest):
                st.markdown(f'<span class="dot dot-green"></span><span style="color:#d0d8e8;font-size:0.82rem">Data: {latest.strftime("%Y-%m-%d")}</span>',
                            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center; padding:12px 0 4px; color:#3a4a66; font-size:0.65rem">
            © 2025 PharmVigil AI · v2.0
        </div>""", unsafe_allow_html=True)

    # ── Apply filters ────────────────────────────────
    f_raw = raw.copy() if has_r else pd.DataFrame()
    f_scores = scores.copy() if has_s else pd.DataFrame()

    if has_r and sel_drugs:
        f_raw = f_raw[f_raw["drug_name"].isin(sel_drugs)]
    if has_r and sel_events:
        f_raw = f_raw[f_raw["adverse_event"].isin(sel_events)]
    if has_r and date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        f_raw = f_raw[(f_raw["report_date"] >= pd.Timestamp(date_range[0])) &
                      (f_raw["report_date"] <= pd.Timestamp(date_range[1]))]
    if has_s and sel_drugs:
        f_scores = f_scores[f_scores["drug_name"].isin(sel_drugs)]
    if has_s and sel_events:
        f_scores = f_scores[f_scores["adverse_event"].isin(sel_events)]

    # ── TABS ─────────────────────────────────────────
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "🔍 Signal Detection",
        "💊 Drug Risk Analysis",
        "📈 Event Trends",
        "🌐 Network Graph",
        "🗺️ Geographic Map",
        "🧪 Risk Prediction",
    ])

    # ══════════════════════════════════════════════════
    #  TAB 1 — SIGNAL DETECTION
    # ══════════════════════════════════════════════════
    with t1:
        if not has_s:
            st.warning("Run the training pipeline first.")
            st.code("python scripts/generate_faers_data.py\npython -m src.models.train_model")
            return

        sig = f_scores[f_scores["risk_score"] >= risk_thresh].copy()

        col_h, col_dl = st.columns([5, 1])
        with col_h:
            st.markdown(f"### 🔍 Safety Signals  <span style='color:#55657d;font-size:0.8rem;font-weight:400'>"
                        f"threshold ≥ {risk_thresh}</span>", unsafe_allow_html=True)
        with col_dl:
            if not sig.empty:
                csv = sig.to_csv(index=False)
                st.download_button("⬇ Export CSV", csv, "safety_signals.csv", "text/csv")

        if sig.empty:
            st.info("No signals above threshold. Lower it in the sidebar.")
        else:
            c1, c2 = st.columns([3, 2])
            with c1:
                top = sig.nlargest(min(20, len(sig)), "risk_score")
                top["label"] = top["drug_name"].str.title() + " → " + top["adverse_event"].str.title()
                fig = go.Figure()
                for _, r in top.iterrows():
                    fig.add_trace(go.Bar(
                        y=[r["label"]], x=[r["risk_score"]], orientation="h",
                        marker=dict(color=ALERT_MAP.get(r.get("alert_level", "low"), C["blue"]), opacity=0.85),
                        hovertemplate=f"<b>{r['drug_name'].title()}</b> → {r['adverse_event'].title()}<br>"
                                     f"Risk: {r['risk_score']:.4f}<br>Alert: {r.get('alert_level','?')}<extra></extra>",
                        showlegend=False,
                    ))
                fig.update_layout(title="Top Safety Signals", xaxis_title="Risk Score",
                                  yaxis=dict(autorange="reversed"), bargap=0.22)
                st.plotly_chart(theme(fig, max(380, len(top) * 26)), use_container_width=True)

            with c2:
                if "alert_level" in sig.columns:
                    ad = sig["alert_level"].value_counts().reset_index()
                    ad.columns = ["Level", "Count"]
                    fig2 = px.pie(ad, values="Count", names="Level", color="Level",
                                 color_discrete_map=ALERT_MAP, hole=0.55)
                    fig2.update_traces(textposition="inside", textinfo="label+percent",
                                       textfont=dict(size=11))
                    fig2.update_layout(title="Alert Distribution",
                                       legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"))
                    st.plotly_chart(theme(fig2, 380), use_container_width=True)

            # ── Severity-coded alert table ───────────
            st.markdown("---")
            st.markdown("### 🚨 Alert Table")
            alert_rows = sig.nlargest(30, "risk_score")
            for _, r in alert_rows.iterrows():
                sev_row(r.get("alert_level", "low"), r["drug_name"],
                        r["adverse_event"], r["risk_score"])

    # ══════════════════════════════════════════════════
    #  TAB 2 — DRUG RISK ANALYSIS
    # ══════════════════════════════════════════════════
    with t2:
        if not has_s:
            st.warning("No signal data.")
            return

        st.markdown("### 💊 Drug × Event Risk Heatmap")

        hm = f_scores.copy()
        top_d = hm.groupby("drug_name")["risk_score"].mean().nlargest(15).index.tolist()
        top_e = hm.groupby("adverse_event")["risk_score"].mean().nlargest(15).index.tolist()
        hm = hm[hm["drug_name"].isin(top_d) & hm["adverse_event"].isin(top_e)]

        if not hm.empty:
            pv = hm.pivot_table(values="risk_score", index="drug_name",
                                columns="adverse_event", aggfunc="mean").fillna(0)
            fig_hm = go.Figure(go.Heatmap(
                z=pv.values, x=[e.title() for e in pv.columns],
                y=[d.title() for d in pv.index],
                colorscale=[[0,"#060a13"],[0.25,"#142042"],[0.5,"#3877f6"],
                            [0.75,"#f5a623"],[1,"#ef4444"]],
                colorbar=dict(title="Risk", thickness=12, len=0.8),
                hovertemplate="<b>%{y}</b> → %{x}<br>Risk: %{z:.4f}<extra></extra>",
            ))
            fig_hm.update_layout(title="Risk Score Matrix",
                                 xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
                                 yaxis=dict(tickfont=dict(size=11)))
            st.plotly_chart(theme(fig_hm, 520), use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 Drug Summary")
        c_bar, c_tbl = st.columns([3, 2])
        with c_bar:
            ds = (f_scores.groupby("drug_name")["risk_score"]
                  .agg(["mean","max","count"]).reset_index()
                  .rename(columns={"mean":"avg_risk","max":"max_risk","count":"signals"})
                  .nlargest(20,"avg_risk"))
            fig_d = go.Figure(go.Bar(
                y=ds["drug_name"].str.title(), x=ds["avg_risk"], orientation="h",
                marker=dict(color=ds["avg_risk"],
                            colorscale=[[0,"#142042"],[0.5,"#f5a623"],[1,"#ef4444"]]),
                hovertemplate="<b>%{y}</b><br>Avg Risk: %{x:.4f}<extra></extra>",
            ))
            fig_d.update_layout(title="Top Drugs by Avg Risk", xaxis_title="Avg Risk Score",
                                yaxis=dict(autorange="reversed"), bargap=0.2)
            st.plotly_chart(theme(fig_d, 480), use_container_width=True)
        with c_tbl:
            st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)
            tbl = ds.copy()
            tbl.columns = ["Drug","Avg Risk","Max Risk","Signals"]
            tbl["Drug"] = tbl["Drug"].str.title()
            tbl = tbl.round(4)
            st.dataframe(tbl, use_container_width=True, hide_index=True, height=440)

    # ══════════════════════════════════════════════════
    #  TAB 3 — ADVERSE EVENT TRENDS
    # ══════════════════════════════════════════════════
    with t3:
        if not has_r or "report_date" not in raw.columns:
            st.warning("No temporal data.")
            return

        st.markdown("### 📈 Reporting Trends")
        td = f_raw.dropna(subset=["report_date"]).copy()
        td["ym"] = td["report_date"].dt.to_period("M").astype(str)

        monthly = td.groupby("ym").size().reset_index(name="count")
        fig_t = go.Figure(go.Scatter(
            x=monthly["ym"], y=monthly["count"], mode="lines+markers",
            line=dict(color=C["blue"], width=2.5), marker=dict(size=4),
            fill="tozeroy", fillcolor="rgba(56,119,246,0.05)",
            hovertemplate="<b>%{x}</b><br>Reports: %{y:,}<extra></extra>",
        ))
        fig_t.update_layout(title="Monthly Report Volume", xaxis_title="Month", yaxis_title="Reports")
        st.plotly_chart(theme(fig_t, 360), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔬 Drug-Specific Trends")
        defs = td["drug_name"].value_counts().head(3).index.tolist()
        sel = st.multiselect("Select drugs", sorted(td["drug_name"].unique()), default=defs, key="t3_drugs")
        if sel:
            palette = [C["blue"], C["emerald"], C["amber"], C["rose"], C["violet"], C["cyan"]]
            sub = td[td["drug_name"].isin(sel)].groupby(["ym","drug_name"]).size().reset_index(name="cnt")
            fig_dm = go.Figure()
            for i, d in enumerate(sel):
                s = sub[sub["drug_name"] == d]
                fig_dm.add_trace(go.Scatter(
                    x=s["ym"], y=s["cnt"], name=d.title(), mode="lines+markers",
                    line=dict(color=palette[i % len(palette)], width=2), marker=dict(size=4),
                ))
            fig_dm.update_layout(title="Drug Trend Comparison", xaxis_title="Month", yaxis_title="Reports",
                                 legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"))
            st.plotly_chart(theme(fig_dm, 380), use_container_width=True)

        if "severity" in td.columns:
            st.markdown("---")
            st.markdown("### 📊 Severity Over Time")
            sv = td.groupby(["ym","severity"]).size().reset_index(name="cnt")
            sc = {"mild":C["low"],"moderate":C["medium"],"severe":C["high"],"life-threatening":C["critical"]}
            fig_sv = px.area(sv, x="ym", y="cnt", color="severity", color_discrete_map=sc,
                             labels={"ym":"Month","cnt":"Reports","severity":"Severity"})
            fig_sv.update_layout(title="Severity Distribution",
                                 legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"))
            st.plotly_chart(theme(fig_sv, 370), use_container_width=True)

    # ══════════════════════════════════════════════════
    #  TAB 4 — NETWORK GRAPH
    # ══════════════════════════════════════════════════
    with t4:
        if not has_s:
            st.warning("No signal data.")
            return

        st.markdown("### 🌐 Drug-Event Relationship Network")
        st.markdown("<p style='color:#6b7fa0;font-size:0.82rem;margin-top:-8px'>"
                    "Interactive graph — <b style='color:#6da0ff'>blue</b> = drugs, "
                    "<b style='color:#f5a623'>amber</b> = events. Edge color = alert level.</p>",
                    unsafe_allow_html=True)

        n_nodes = st.slider("Number of top signal pairs", 10, 80, 35, key="net_n")
        with st.spinner("Building network …"):
            html = build_network_html(f_scores, top_n=n_nodes)
        components.html(html, height=520, scrolling=False)

    # ══════════════════════════════════════════════════
    #  TAB 5 — GEOGRAPHIC MAP
    # ══════════════════════════════════════════════════
    with t5:
        if not has_r or "country" not in raw.columns:
            st.warning("No geographic data.")
            return

        st.markdown("### 🗺️ Geographic Distribution of Reports")

        iso = {
            "US":"USA","UK":"GBR","DE":"DEU","FR":"FRA","JP":"JPN","CA":"CAN",
            "AU":"AUS","BR":"BRA","IN":"IND","MX":"MEX","IT":"ITA","ES":"ESP",
            "KR":"KOR","CN":"CHN","RU":"RUS","ZA":"ZAF","AR":"ARG","NL":"NLD",
            "SE":"SWE","CH":"CHE","BE":"BEL","AT":"AUT",
        }
        geo = f_raw["country"].value_counts().reset_index()
        geo.columns = ["code","reports"]
        geo["iso"] = geo["code"].map(iso).fillna(geo["code"])

        fig_geo = px.choropleth(
            geo, locations="iso", color="reports", hover_name="code",
            color_continuous_scale=[[0,"#0d1320"],[0.3,"#142042"],[0.6,"#3877f6"],[1,"#f43f5e"]],
            labels={"reports":"Reports"}, title="Reports by Country",
        )
        fig_geo.update_layout(
            geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)",
                     landcolor="#111827", showframe=False, coastlinecolor="#1c2740"),
        )
        st.plotly_chart(theme(fig_geo, 500), use_container_width=True)

        st.markdown("### 📊 Country Breakdown")
        st.dataframe(geo[["code","reports"]].rename(columns={"code":"Country","reports":"Reports"}),
                     use_container_width=True, hide_index=True, height=300)

    # ══════════════════════════════════════════════════
    #  TAB 6 — RISK PREDICTION
    # ══════════════════════════════════════════════════
    with t6:
        st.markdown("### 🧪 Real-Time Risk Prediction")
        st.markdown("<p style='color:#6b7fa0;font-size:0.82rem;margin-top:-8px'>"
                    "Submit a drug–event pair to the AI model for instant risk assessment.</p>",
                    unsafe_allow_html=True)

        c_in, c_out = st.columns([2, 3])
        with c_in:
            st.markdown('<div class="panel"><h4>Input</h4>', unsafe_allow_html=True)
            p_drug = st.selectbox("💊 Drug", [""] + (sorted(raw["drug_name"].dropna().unique().tolist()) if has_r else []),
                                  key="pred_d")
            p_event = st.selectbox("⚠️ Adverse Event",
                                   [""] + (sorted(raw["adverse_event"].dropna().unique().tolist()) if has_r else []),
                                   key="pred_e")
            go_btn = st.button("🔍  Analyze Risk", use_container_width=True, type="primary")
            st.markdown("</div>", unsafe_allow_html=True)

        with c_out:
            if go_btn and p_drug and p_event:
                with st.spinner("🔄 Running inference …"):
                    result, ms = api_predict(p_drug, p_event)

                if result:
                    risk = result.get("risk_score", 0)
                    sig_str = result.get("signal_strength", 0)
                    alert = result.get("alert_level", "low")
                    ac = ALERT_MAP.get(alert, C["blue"])

                    r1, r2, r3 = st.columns(3)
                    with r1:
                        st.markdown(f'<div class="pred-card"><div class="pred-val" style="color:{ac}">{risk:.4f}</div>'
                                    f'<div class="pred-lbl">Risk Score</div></div>', unsafe_allow_html=True)
                    with r2:
                        st.markdown(f'<div class="pred-card"><div class="pred-val" style="color:{C["cyan"]}">{sig_str:.4f}</div>'
                                    f'<div class="pred-lbl">Signal Strength</div></div>', unsafe_allow_html=True)
                    with r3:
                        st.markdown(f'<div class="pred-card"><div class="pred-val" style="color:{ac};font-size:1.8rem">{alert.upper()}</div>'
                                    f'<div class="pred-lbl">Alert Level</div></div>', unsafe_allow_html=True)

                    # Gauge
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number", value=risk,
                        number=dict(font=dict(size=32, color=C["text"]), valueformat=".4f"),
                        gauge=dict(
                            axis=dict(range=[0,1], tickwidth=1, tickcolor=C["text_muted"]),
                            bar=dict(color=ac, thickness=0.25),
                            bgcolor="rgba(13,19,32,0.5)", borderwidth=1, bordercolor=C["border"],
                            steps=[
                                dict(range=[0,0.3], color="rgba(34,197,94,0.06)"),
                                dict(range=[0.3,0.6], color="rgba(234,179,8,0.06)"),
                                dict(range=[0.6,0.8], color="rgba(249,115,22,0.06)"),
                                dict(range=[0.8,1], color="rgba(239,68,68,0.06)"),
                            ],
                            threshold=dict(line=dict(color=C["rose"],width=3), thickness=0.8, value=risk),
                        ),
                        title=dict(text=f"{p_drug.title()} + {p_event.title()}", font=dict(size=13)),
                    ))
                    st.plotly_chart(theme(fig_g, 280), use_container_width=True)

                    st.markdown(f"<p style='text-align:center;color:#55657d;font-size:0.72rem'>"
                                f"API latency: {ms:.0f}ms</p>", unsafe_allow_html=True)
                else:
                    st.error("Could not reach the API. Ensure the FastAPI server is running on localhost:8000.")
                    st.code("uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload")

            elif go_btn:
                st.warning("Select both a drug and event.")
            else:
                st.markdown("""
                <div class="panel" style="text-align:center; padding:50px 24px;">
                    <p style="font-size:2.5rem; margin-bottom:8px">🧬</p>
                    <h3 style="margin-bottom:6px !important">Ready to Analyze</h3>
                    <p style="color:#55657d; font-size:0.82rem">
                        Select a drug and adverse event, then click <b>Analyze Risk</b>.
                    </p>
                </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
