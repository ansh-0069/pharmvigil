"""
AI Pharmacovigilance Intelligence Platform — Enterprise Dashboard v2.0

Professional-grade analytics dashboard for pharmaceutical safety monitoring.
Features: signal detection, drug risk heatmaps, network graph, geographic
risk map, adverse event trends, real-time prediction, system monitoring,
and CSV export capabilities.
"""
from __future__ import annotations

import html
import io
import json
import os
import re
import subprocess
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure repository root is importable on hosted environments (e.g., Streamlit Cloud).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from capa_board import render_capa_board
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

def resolve_api_base() -> str:
    secret_api_base = ""
    try:
        secret_api_base = st.secrets.get("PHARMVIGIL_API_BASE", "")
    except Exception:
        pass

    return os.getenv("PHARMVIGIL_API_BASE", secret_api_base or "http://localhost:8000").rstrip("/")


API_BASE = resolve_api_base()
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


APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --app-bg: #0b111b;
    --app-bg-soft: #121a28;
    --app-panel: #0f1724;
    --app-line: #334155;
    --app-line-soft: rgba(71, 85, 105, 0.45);
    --app-text: #e2e8f0;
    --app-muted: #94a3b8;
    --app-accent: #7dd3fc;
}

/* Hide Streamlit chrome */
header[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"],
#MainMenu,
footer,
section[data-testid="stSidebar"] {
    display: none !important;
}

/* Full bleed canvas */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background:
        radial-gradient(1100px 560px at 2% -15%, rgba(56, 189, 248, 0.14), transparent 58%),
        radial-gradient(850px 420px at 98% 3%, rgba(14, 165, 233, 0.1), transparent 62%),
        linear-gradient(180deg, #0b111b 0%, #0c1320 42%, #0a1019 100%) !important;
    color: var(--app-text) !important;
    font-family: 'Manrope', sans-serif !important;
}

.block-container { max-width: 100vw !important; padding: 0.5rem 3.8vw 7rem !important; }

/* Typography direction */
h1, h2, h4, h5 {
    color: var(--app-text) !important;
    letter-spacing: -0.03em;
}

h1, .hero-title {
    font-family: 'Space Grotesk', sans-serif !important;
}

h3 {
    color: #94A3B8 !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    margin-bottom: 0.7rem !important;
}

/* Hero refresh */
.hero {
    background: linear-gradient(130deg, rgba(15, 23, 36, 0.9), rgba(13, 20, 31, 0.88));
    border: 1px solid var(--app-line-soft);
    border-radius: 26px;
    padding: clamp(1.3rem, 2vw, 2.1rem) clamp(1.1rem, 3vw, 2.6rem);
    margin: 0.4rem 0 1.2rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(8px);
}

.hero::after {
    content: '';
    position: absolute;
    right: -120px;
    top: -90px;
    width: 340px;
    height: 260px;
    background: radial-gradient(circle, rgba(125, 211, 252, 0.2), transparent 68%);
}

.hero-title {
    display: flex;
    align-items: center;
    gap: 0.42em;
    line-height: 1.04;
    color: #f1f5f9 !important;
    margin-bottom: 0.8rem;
    max-width: 100%;
    white-space: nowrap;
    text-wrap: nowrap;
}

.hero-title-icon {
    font-size: clamp(1.35rem, 2.8vw, 4rem);
    line-height: 1;
    flex: 0 0 auto;
}

.hero-title-text {
    font-size: clamp(1.2rem, 3.05vw, 3.6rem);
    letter-spacing: -0.02em;
    flex: 1 1 auto;
    min-width: 0;
}

.hero-sub {
    font-size: clamp(0.9rem, 1.35vw, 1.15rem);
    color: #a8b4c8;
    max-width: 68ch;
}

.hero-badge {
    border-radius: 999px;
    border: 1px solid rgba(125, 211, 252, 0.32);
    color: #b8ebff;
    background: rgba(56, 189, 248, 0.14);
}

/* Existing cards upgraded */
.kpi, .panel, .pred-card {
    border-radius: 16px !important;
    border: 1px solid var(--app-line-soft) !important;
    background: linear-gradient(180deg, rgba(15, 23, 36, 0.86), rgba(12, 18, 28, 0.85)) !important;
    box-shadow: none !important;
}

.kpi {
    padding: 1.05rem 0.95rem;
    min-height: 108px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.kpi-icon {
    font-size: 20px;
    line-height: 1;
    margin-bottom: 0.45rem;
}

.kpi-val {
    font-size: clamp(1.15rem, 1.55vw, 1.95rem);
    line-height: 1.1;
}

.kpi-lbl {
    font-size: 0.74rem;
}

.pred-card {
    padding: 1rem 0.95rem;
    min-height: 104px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.pred-val {
    font-size: clamp(1.25rem, 1.95vw, 2.6rem);
    line-height: 1.04;
}

.kpi-val,
.pred-val,
.sev-score {
    font-family: 'IBM Plex Mono', monospace !important;
}

.panel h4,
.kpi-lbl,
.pred-lbl {
    color: var(--app-muted) !important;
    letter-spacing: 0.12em !important;
}

.kpi-icon,
.sev-drug,
.sev-label,
.pred-val,
.kpi-val {
    color: #e2e8f0 !important;
}

.panel span,
.panel p,
.panel div {
    color: #b8c2d3 !important;
}

.sev-row {
    background: rgba(15, 23, 36, 0.65) !important;
    border-left-width: 4px;
}

/* Tabs become premium segmented rail */
.stTabs [data-baseweb="tab-list"] {
    padding: 0.4rem;
    border-radius: 16px;
    border: 1px solid var(--app-line-soft);
    background: rgba(15, 23, 36, 0.62);
    backdrop-filter: blur(8px);
    overflow-x: auto;
    overflow-y: hidden;
    flex-wrap: nowrap;
    scrollbar-width: none;
}

.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
    display: none;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    color: #8ea0ba;
    font-weight: 600;
    font-size: 0.86rem;
    padding: 0.58rem 0.9rem;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(120deg, rgba(14, 165, 233, 0.2), rgba(56, 189, 248, 0.24)) !important;
    color: #d5f3ff !important;
    border: 1px solid rgba(56, 189, 248, 0.4);
}

/* Native widgets re-skin */
.stButton > button,
.stDownloadButton > button {
    border-radius: 999px !important;
    border: 1px solid rgba(56, 189, 248, 0.45) !important;
    background: linear-gradient(120deg, rgba(14, 116, 144, 0.38), rgba(14, 165, 233, 0.2)) !important;
    color: #d9f5ff !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    border-color: rgba(125, 211, 252, 0.88) !important;
    background: linear-gradient(120deg, rgba(14, 116, 144, 0.52), rgba(14, 165, 233, 0.32)) !important;
}

.stSelectbox [data-baseweb="select"],
.stMultiSelect [data-baseweb="select"],
.stTextInput input,
.stDateInput input,
.stTextArea textarea,
.stSlider [data-baseweb="slider"] {
    border-radius: 14px !important;
    border-color: rgba(71, 85, 105, 0.6) !important;
    background: rgba(15, 23, 36, 0.9) !important;
    color: #d9e2ef !important;
}

.stMultiSelect [data-baseweb="tag"] {
    background: rgba(30, 41, 59, 0.9) !important;
    color: #c8d4e7 !important;
    border: 1px solid rgba(71, 85, 105, 0.8) !important;
}

[data-testid="stDataFrame"],
[data-testid="stMetric"],
[data-testid="stPlotlyChart"],
.stAlert {
    border-radius: 12px !important;
    border: 1px solid var(--app-line-soft) !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* Plotly charts should float directly on the page */
[data-testid="stPlotlyChart"] > div,
[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plot-container,
[data-testid="stPlotlyChart"] .svg-container {
    background: transparent !important;
    box-shadow: none !important;
}

/* Minimalist financial-ledger dataframe styling */
[data-testid="stDataFrame"] [role="grid"],
[data-testid="stTable"] table {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stDataFrame"] [role="columnheader"],
[data-testid="stTable"] thead th {
    background: transparent !important;
    color: #94A3B8 !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid rgba(51, 65, 85, 0.8) !important;
    border-right: none !important;
}

[data-testid="stDataFrame"] [role="gridcell"],
[data-testid="stTable"] tbody td {
    background: transparent !important;
    color: #d9e2ef !important;
    border-bottom: 1px solid rgba(51, 65, 85, 0.45) !important;
    border-right: none !important;
    font-size: 12px !important;
}

[data-testid="stDataFrame"] [role="row"]:hover [role="gridcell"],
[data-testid="stTable"] tbody tr:hover td {
    background: rgba(30, 41, 59, 0.28) !important;
}

/* Flat metric cards */
[data-testid="stMetric"] {
    background: #121a28 !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 0.8rem 0.9rem !important;
    box-shadow: none !important;
}

[data-testid="stMetric"] * {
    box-shadow: none !important;
}

[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    font-size: 11px !important;
}

[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Floating command menu and nav (safe layout version) */
.cmd-anchor,
.nav-anchor {
    display: none;
}

.app-nav {
    display: flex;
    gap: 0.35rem;
    align-items: center;
    flex-wrap: wrap;
    position: sticky;
    top: 0.35rem;
    z-index: 20;
    margin: 0.25rem 0 0.7rem;
    width: fit-content;
    padding: 0.48rem 0.72rem;
    border-radius: 999px;
    border: 1px solid var(--app-line-soft);
    background: rgba(15, 23, 36, 0.72);
    backdrop-filter: blur(10px);
}

.app-nav a {
    color: #b9c7da;
    text-decoration: none;
    border-radius: 999px;
    padding: 0.34rem 0.7rem;
    border: 1px solid rgba(71, 85, 105, 0.7);
    font-size: 0.72rem;
    font-weight: 700;
}

.app-nav a:hover {
    color: #d9f5ff;
    border-color: rgba(56, 189, 248, 0.75);
    background: rgba(14, 165, 233, 0.2);
}

.cmd-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    color: #94A3B8;
    letter-spacing: 0.16em;
    margin: 0.45rem 0;
}

/* Command menu monitor block */
.sys-monitor {
    border: 1px solid rgba(71, 85, 105, 0.55);
    border-radius: 14px;
    background: rgba(15, 23, 36, 0.65);
    padding: 0.85rem 0.95rem;
    margin-top: 0.5rem;
    width: 100%;
    box-sizing: border-box;
}

.sys-monitor-title {
    color: #94A3B8;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
    margin-bottom: 0.65rem;
}

.sys-monitor-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 0.36rem 0;
    color: #d9e2ef;
    font-size: 0.82rem;
    border-bottom: 1px solid rgba(51, 65, 85, 0.45);
}

.sys-monitor-row:last-child {
    border-bottom: none;
}

.sys-monitor-left {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #c7d2e2;
}

.sys-monitor-right {
    color: #8ea0ba;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    white-space: nowrap;
}

.stCode, .stMarkdown code {
    background: rgba(30, 41, 59, 0.6) !important;
    color: #b7c6da !important;
}

/* Scroll reveal animation */
.reveal-item {
    opacity: 0;
    transform: translate3d(0, 28px, 0) scale(0.985);
    transition: opacity 0.9s cubic-bezier(0.2, 0.65, 0.2, 1), transform 0.9s cubic-bezier(0.2, 0.65, 0.2, 1);
    will-change: opacity, transform;
}

.reveal-item.reveal-live {
    opacity: 1;
    transform: translate3d(0, 0, 0) scale(1);
}

@media (max-width: 980px) {
    .block-container {
        padding: 0.25rem 0.85rem 4.5rem !important;
    }
    .app-nav {
        width: 100%;
        border-radius: 16px;
    }
    .sys-monitor {
        padding: 0.65rem 0.72rem;
        border-radius: 12px;
    }
    .sys-monitor-title {
        margin-bottom: 0.45rem;
    }
    .sys-monitor-row {
        padding: 0.24rem 0;
        font-size: 0.78rem;
    }
    .sys-monitor-right {
        font-size: 0.72rem;
    }
    .hero-title {
        line-height: 1.03;
        max-width: 100%;
        gap: 0.3em;
    }
    .hero-title-icon {
        font-size: clamp(1rem, 6vw, 1.9rem);
    }
    .hero-title-text {
        font-size: clamp(0.78rem, 3.45vw, 1.65rem);
    }
    .hero-sub {
        font-size: 0.97rem;
    }
    .kpi {
        min-height: 94px;
    }
    .pred-card {
        min-height: 90px;
        padding: 0.85rem 0.8rem;
    }
}
</style>
"""


SCROLL_OBSERVER_JS = """
<script>
(function () {
  const doc = window.parent.document;
  if (!doc || doc.documentElement.dataset.scrollyReady === '1') return;
  doc.documentElement.dataset.scrollyReady = '1';

  const pickTargets = () => {
    const targets = [];
    const selectors = [
      '[data-testid="stPlotlyChart"]',
      '[data-testid="stDataFrame"]',
      '[data-testid="stMetric"]',
      '[data-testid="stAlert"]',
      '.hero', '.kpi', '.panel', '.pred-card', '.sev-row', '.stTabs'
    ];

    selectors.forEach((sel) => {
      doc.querySelectorAll(sel).forEach((el) => {
        if (!el.closest('section[data-testid="stSidebar"]')) targets.push(el);
      });
    });

    doc.querySelectorAll('[data-testid="stVerticalBlock"] > div').forEach((el) => {
      if (!el.querySelector('[data-testid="stHorizontalBlock"]') && !el.closest('section[data-testid="stSidebar"]')) {
        targets.push(el);
      }
    });

    return [...new Set(targets)].filter((el) => el.offsetHeight > 20);
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('reveal-live');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.14, rootMargin: '0px 0px -8% 0px' });

  const arm = () => {
    pickTargets().forEach((el) => {
      if (!el.classList.contains('reveal-item')) {
        el.classList.add('reveal-item');
        observer.observe(el);
      }
    });
  };

  arm();
  const mo = new MutationObserver(() => arm());
  mo.observe(doc.body, { childList: true, subtree: true });
})();
</script>
"""


NAV_TABS_JS = """
<script>
(function () {
    const doc = window.parent.document;
    if (!doc || doc.documentElement.dataset.navTabsReady === '1') return;
    doc.documentElement.dataset.navTabsReady = '1';

    const bindNav = () => {
        const nav = doc.querySelector('.app-nav');
        if (!nav || nav.dataset.bound === '1') return;
        nav.dataset.bound = '1';

        nav.addEventListener('click', (event) => {
            const link = event.target.closest('a[data-tab-index]');
            if (!link) return;
            event.preventDefault();

            const idx = Number(link.getAttribute('data-tab-index'));
            if (Number.isNaN(idx)) return;

            const tabButtons = doc.querySelectorAll('.stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]');
            if (idx < 0 || idx >= tabButtons.length) return;

            tabButtons[idx].click();
            const tabsRoot = doc.querySelector('.stTabs');
            if (tabsRoot) {
                tabsRoot.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    };

    bindNav();
    const mo = new MutationObserver(() => bindNav());
    mo.observe(doc.body, { childList: true, subtree: true });
})();
</script>
"""


# ═══════════════════════════════════════════════════════
#  CSS INJECTION — Enterprise Dark Theme
# ═══════════════════════════════════════════════════════
def inject_css():
    st.markdown(APP_CSS, unsafe_allow_html=True)


def inject_scroll_observer():
    components.html(SCROLL_OBSERVER_JS, height=0, width=0)


def inject_nav_tab_clicks():
    components.html(NAV_TABS_JS, height=0, width=0)


def render_shell_navigation():
    st.markdown('<div class="nav-anchor"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="app-nav">
            <a href="#" data-tab-index="0">Monitor</a>
            <a href="#" data-tab-index="1">Investigate</a>
            <a href="#" data-tab-index="2">Inbox</a>
            <a href="#" data-tab-index="3">Operations</a>
            <a href="#" data-tab-index="4">Activity</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_command_menu(raw: pd.DataFrame, scores: pd.DataFrame, has_r: bool, has_s: bool):
    st.markdown('<div class="cmd-anchor"></div>', unsafe_allow_html=True)
    st.markdown('<div class="cmd-title">Global Filter Bar</div>', unsafe_allow_html=True)

    filter_row_1 = st.columns([1.3, 1.3, 1, 1, 1.2], gap="small")
    drug_opts = sorted(raw["drug_name"].dropna().unique().tolist()) if has_r and "drug_name" in raw.columns else []
    event_opts = sorted(raw["adverse_event"].dropna().unique().tolist()) if has_r and "adverse_event" in raw.columns else []
    severity_opts = sorted(raw["severity"].dropna().unique().tolist()) if has_r and "severity" in raw.columns else []
    country_opts = sorted(raw["country"].dropna().unique().tolist()) if has_r and "country" in raw.columns else []
    alert_opts = sorted(scores["alert_level"].dropna().unique().tolist()) if has_s and "alert_level" in scores.columns else []

    with filter_row_1[0]:
        sel_drugs = st.multiselect("Drug Filters", drug_opts, default=[], key="sb_drugs")
    with filter_row_1[1]:
        sel_events = st.multiselect("Event Filters", event_opts, default=[], key="sb_events")
    with filter_row_1[2]:
        sel_severity = st.multiselect("Severity", severity_opts, default=[], key="sb_severity")
    with filter_row_1[3]:
        sel_country = st.multiselect("Country", country_opts, default=[], key="sb_country")
    with filter_row_1[4]:
        sel_alert = st.multiselect("Alert Level", alert_opts, default=[], key="sb_alert")

    filter_row_2 = st.columns([1.1, 1.6, 0.8], gap="small")
    with filter_row_2[0]:
        risk_thresh = st.slider("Risk Threshold", 0.0, 1.0, 0.6, 0.05, key="sb_thresh")

    if has_r and "report_date" in raw.columns:
        mn, mx = raw["report_date"].min(), raw["report_date"].max()
        if pd.notna(mn) and pd.notna(mx):
            with filter_row_2[1]:
                date_range = st.date_input(
                    "Date Window",
                    value=(mn.date(), mx.date()),
                    min_value=mn.date(),
                    max_value=mx.date(),
                    key="sb_date",
                )
        else:
            date_range = None
    else:
        date_range = None

    with filter_row_2[2]:
        if st.button("Reset Filters", use_container_width=True, key="reset_filters"):
            for key in ["sb_drugs", "sb_events", "sb_severity", "sb_country", "sb_alert"]:
                st.session_state[key] = []
            st.session_state["sb_thresh"] = 0.6
            if has_r and "report_date" in raw.columns and pd.notna(raw["report_date"].min()) and pd.notna(raw["report_date"].max()):
                st.session_state["sb_date"] = (raw["report_date"].min().date(), raw["report_date"].max().date())
            st.rerun()

    api_ok, api_ms = api_health()
    dot = "dot-green" if api_ok else "dot-red"
    label = "Operational" if api_ok else "Offline"

    model_exists = Path("models/xgboost_classifier.joblib").exists()
    mdot = "dot-green" if model_exists else "dot-amber"
    mlbl = "Loaded" if model_exists else "Not trained"

    latest_text = "Unknown"
    if has_r and "report_date" in raw.columns:
        latest = raw["report_date"].max()
        if pd.notna(latest):
            latest_text = latest.strftime("%Y-%m-%d")

    monitor_html = f"""
    <div class="sys-monitor">
        <div class="sys-monitor-title">System Monitor</div>
        <div class="sys-monitor-row">
            <span class="sys-monitor-left"><span class="dot {dot}"></span>API: {label}</span>
            <span class="sys-monitor-right">{api_ms:.0f}ms</span>
        </div>
        <div class="sys-monitor-row">
            <span class="sys-monitor-left"><span class="dot {mdot}"></span>Model: {mlbl}</span>
            <span class="sys-monitor-right">ready</span>
        </div>
        <div class="sys-monitor-row">
            <span class="sys-monitor-left"><span class="dot dot-green"></span>Data: {latest_text}</span>
            <span class="sys-monitor-right">latest</span>
        </div>
    </div>
    """
    st.markdown(monitor_html, unsafe_allow_html=True)

    return {
        "drugs": sel_drugs,
        "events": sel_events,
        "severity": sel_severity,
        "country": sel_country,
        "alert_levels": sel_alert,
        "risk_threshold": risk_thresh,
        "date_range": date_range,
    }


@st.cache_data(ttl=30)
def fetch_events(limit: int = 50) -> list:
    try:
        r = requests.get(f"{API_BASE}/api/v1/events/recent?limit={limit}", timeout=5)
        return r.json() if r.status_code == 200 else []
    except requests.RequestException:
        return []

@st.cache_data(ttl=30)
def fetch_open_capas() -> list:
    try:
        r = requests.get(f"{API_BASE}/capa/open", timeout=5)
        return r.json() if r.status_code == 200 else []
    except requests.RequestException:
        return []

@st.cache_data(ttl=30)
def fetch_overdue_submissions() -> list:
    try:
        r = requests.get(f"{API_BASE}/submissions/overdue", timeout=5)
        return r.json() if r.status_code == 200 else []
    except requests.RequestException:
        return []

@st.cache_data(ttl=60)
def fetch_audit_portfolio() -> list:
    try:
        r = requests.get(f"{API_BASE}/audit/portfolio", timeout=5)
        return r.json() if r.status_code == 200 else []
    except requests.RequestException:
        return []


@st.cache_data(ttl=30)
def fetch_submissions() -> list:
    try:
        r = requests.get(f"{API_BASE}/submissions", timeout=5)
        return r.json() if r.status_code == 200 else []
    except requests.RequestException:
        return []


@st.cache_data(ttl=30)
def fetch_entity_events(entity_type: str, entity_id: str) -> list:
    try:
        r = requests.get(f"{API_BASE}/api/v1/events/{entity_type}/{entity_id}", timeout=5)
        return r.json() if r.status_code == 200 else []
    except requests.RequestException:
        return []


def create_submission(product_id: str, submission_type: str, due_date: datetime, complexity_weight: float, notes: str) -> tuple[bool, Optional[str]]:
    try:
        r = requests.post(
            f"{API_BASE}/submissions",
            json={
                "product_id": product_id,
                "submission_type": submission_type,
                "due_date": due_date.isoformat(),
                "complexity_weight": complexity_weight,
                "notes": notes or None,
            },
            timeout=10,
        )
        if r.status_code == 201:
            fetch_submissions.clear()
            fetch_overdue_submissions.clear()
            fetch_events.clear()
            return True, None
        detail = r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json") else None
        return False, detail or f"Submission API returned {r.status_code}."
    except requests.RequestException as exc:
        return False, str(exc)


def update_submission(submission_id: int, status: str) -> tuple[bool, Optional[str]]:
    payload = {"status": status}
    if status.upper() == "SUBMITTED":
        payload["submitted_date"] = datetime.utcnow().isoformat()
    try:
        r = requests.patch(f"{API_BASE}/submissions/{submission_id}/status", json=payload, timeout=10)
        if r.status_code == 200:
            fetch_submissions.clear()
            fetch_overdue_submissions.clear()
            fetch_events.clear()
            return True, None
        detail = r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json") else None
        return False, detail or f"Status update failed with {r.status_code}."
    except requests.RequestException as exc:
        return False, str(exc)


def create_capa_case(product_id: str, title: str, description: str, priority: str, assigned_to: str = "", due_date: Optional[datetime] = None) -> tuple[bool, Optional[str]]:
    payload = {
        "product_id": product_id,
        "title": title,
        "description": description or None,
        "priority": priority,
        "assigned_to": assigned_to or None,
        "due_date": due_date.isoformat() if due_date else None,
    }
    try:
        r = requests.post(f"{API_BASE}/capa", json=payload, timeout=10)
        if r.status_code == 201:
            fetch_open_capas.clear()
            fetch_events.clear()
            return True, None
        detail = r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json") else None
        return False, detail or f"CAPA API returned {r.status_code}."
    except requests.RequestException as exc:
        return False, str(exc)


def normalize_event_description(description: str) -> str:
    """Convert legacy HTML-like audit text into clean plain text for timeline rendering."""
    if not description:
        return ""
    text = html.unescape(description)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    if len(text) > 320:
        text = text[:317] + "..."
    return text


def _build_quick_scores(raw_csv: Path, out_csv: Path) -> None:
    """Generate lightweight signal scores for demo environments without training."""
    df = pd.read_csv(raw_csv)
    if df.empty or "drug_name" not in df.columns or "adverse_event" not in df.columns:
        raise ValueError("Raw dataset is missing required columns.")

    work = df.copy()
    work["drug_name"] = work["drug_name"].astype(str).str.lower().str.strip()
    work["adverse_event"] = work["adverse_event"].astype(str).str.lower().str.strip()
    work = work[(work["drug_name"] != "") & (work["adverse_event"] != "")]

    total_reports = max(len(work), 1)

    pair = (
        work.groupby(["drug_name", "adverse_event"]).size().reset_index(name="co_occurrence_count")
    )
    drug_freq = work.groupby("drug_name").size().rename("drug_frequency")
    event_freq = work.groupby("adverse_event").size().rename("event_frequency")

    pair = pair.merge(drug_freq, on="drug_name", how="left")
    pair = pair.merge(event_freq, on="adverse_event", how="left")
    pair["total_reports"] = total_reports

    # Simple disproportionality approximations for demo-only rendering.
    pair["prr"] = (pair["co_occurrence_count"] / pair["drug_frequency"].clip(lower=1)) / (
        pair["event_frequency"] / total_reports
    ).clip(lower=1e-6)
    pair["ror"] = ((pair["co_occurrence_count"] + 0.5) / (pair["drug_frequency"] - pair["co_occurrence_count"] + 0.5).clip(lower=0.5)) / (
        (pair["event_frequency"] + 0.5) / (total_reports - pair["event_frequency"] + 0.5).clip(lower=0.5)
    )

    pair["log_prr"] = np.log1p(pair["prr"].clip(lower=0))
    pair["log_ror"] = np.log1p(pair["ror"].clip(lower=0))
    pair["drug_event_ratio"] = pair["co_occurrence_count"] / pair["drug_frequency"].clip(lower=1)
    pair["time_trend_slope"] = 0.0

    # Deterministic blended demo score in [0, 1].
    a = pair["co_occurrence_count"] / pair["co_occurrence_count"].max()
    b = pair["drug_event_ratio"].clip(upper=1)
    c = (pair["log_prr"] / pair["log_prr"].max()).fillna(0)
    pair["risk_score"] = (0.45 * a + 0.35 * b + 0.20 * c).clip(0, 1).round(4)
    pair["signal_strength"] = pair["risk_score"]

    pair["alert_level"] = "low"
    pair.loc[pair["risk_score"] >= 0.30, "alert_level"] = "medium"
    pair.loc[pair["risk_score"] >= 0.60, "alert_level"] = "high"
    pair.loc[pair["risk_score"] >= 0.80, "alert_level"] = "critical"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pair.to_csv(out_csv, index=False)


def bootstrap_demo_assets() -> None:
    """Create demo data/train artifacts when running in a clean deployment."""
    raw_missing = not RAW_PATH.exists()
    scores_missing = not SCORES_PATH.exists()
    model_missing = not Path("models/xgboost_classifier.joblib").exists()

    if not raw_missing and not scores_missing and not model_missing:
        return

    missing_items = []
    if raw_missing:
        missing_items.append(str(RAW_PATH))
    if scores_missing:
        missing_items.append(str(SCORES_PATH))
    if model_missing:
        missing_items.append("models/xgboost_classifier.joblib")

    st.warning(
        "Deployment is missing artifacts. Use Quick Demo Mode for instant charts, "
        "or Full Initialize for trained models.",
        icon="⚙️",
    )
    st.caption("Missing: " + ", ".join(missing_items))

    repo_root = Path(__file__).resolve().parents[1]
    c_quick, c_full = st.columns(2)

    with c_quick:
        if st.button("Quick Demo Mode (Instant)", key="quick_demo_assets", use_container_width=True):
            try:
                if raw_missing:
                    with st.spinner("Generating synthetic dataset..."):
                        subprocess.run(
                            [sys.executable, "scripts/generate_faers_data.py"],
                            cwd=repo_root,
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                with st.spinner("Building lightweight demo scores..."):
                    _build_quick_scores(RAW_PATH, SCORES_PATH)
                st.success("Quick demo assets are ready. Reloading dashboard...")
                st.cache_data.clear()
                st.rerun()
            except (subprocess.CalledProcessError, ValueError) as exc:
                st.error(f"Quick demo setup failed: {str(exc)[:600]}")

    with c_full:
        if st.button("Full Initialize (Train Models)", type="primary", key="init_demo_assets", use_container_width=True):
            try:
                with st.spinner("Generating synthetic dataset..."):
                    subprocess.run(
                        [sys.executable, "scripts/generate_faers_data.py"],
                        cwd=repo_root,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                with st.spinner("Training models and creating scores..."):
                    subprocess.run(
                        [sys.executable, "-m", "src.models.train_model"],
                        cwd=repo_root,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                st.success("Full initialization complete. Reloading dashboard...")
                st.cache_data.clear()
                st.rerun()
            except subprocess.CalledProcessError as exc:
                stderr_text = (exc.stderr or "").strip()
                if stderr_text:
                    st.error(f"Full initialization failed: {stderr_text[:600]}")
                else:
                    st.error("Full initialization failed. Check deployment logs for details.")


def _local_predict(drug: str, event: str) -> Optional[dict]:
    """Fallback predictor for environments where the FastAPI service is unavailable."""
    try:
        from src.models.predict import get_predictor
        pred = get_predictor().predict(drug_name=drug, adverse_event=event)
        return {
            "drug_name": pred.drug_name,
            "adverse_event": pred.adverse_event,
            "risk_score": pred.risk_score,
            "signal_strength": pred.signal_strength,
            "alert_level": pred.alert_level,
        }
    except Exception:
        return None


# ═══════════════════════════════════════════════════════
_PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Manrope, system-ui, sans-serif", color="#d9e2ef", size=12),
    margin=dict(l=16, r=16, t=44, b=16),
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        tickfont=dict(color="#cbd5e1", size=11),
        titlefont=dict(color="#94a3b8", size=11),
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        tickfont=dict(color="#cbd5e1", size=11),
        titlefont=dict(color="#94a3b8", size=11),
    ),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11)),
    hoverlabel=dict(bgcolor="rgba(15,23,36,0.98)", font_size=12, bordercolor="#334155"),
)

def theme(fig: go.Figure, h: int = 420) -> go.Figure:
    fig.update_layout(**_PLOTLY_BASE, height=h)
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False)
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


def api_predict(drug: str, event: str) -> tuple[Optional[dict], float, Optional[str]]:
    """Returns (result_dict, response_time_ms, error_message)."""
    t0 = time.time()
    try:
        r = requests.post(f"{API_BASE}/predict",
                          json={"drug_name": drug, "adverse_event": event}, timeout=30)
        elapsed = (time.time() - t0) * 1000
        if r.status_code == 200:
            return r.json(), elapsed, None
        msg = f"Prediction API returned {r.status_code}."
        try:
            detail = r.json().get("detail")
            if detail:
                msg = f"{msg} {detail}"
        except Exception:
            pass
        local = _local_predict(drug, event)
        if local is not None:
            return local, elapsed, None
        return None, elapsed, msg
    except (requests.ConnectionError, requests.Timeout, requests.exceptions.RequestException):
        elapsed = (time.time() - t0) * 1000
    local = _local_predict(drug, event)
    if local is not None:
        return local, elapsed, None
    return None, elapsed, f"Could not connect to the API server at {API_BASE}."


def api_health() -> tuple[bool, float]:
    """Ping /health and return (is_healthy, ms)."""
    t0 = time.time()
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        ms = (time.time() - t0) * 1000
        return r.status_code == 200, ms
    except Exception:
        ms = (time.time() - t0) * 1000
        # Deployed Streamlit often runs without local FastAPI; treat local model availability as healthy.
        model_ready = Path("models/xgboost_classifier.joblib").exists()
        return model_ready, ms


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


def apply_global_filters(raw: pd.DataFrame, scores: pd.DataFrame, filters: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    f_raw = raw.copy()
    f_scores = scores.copy()

    if not f_raw.empty and filters["drugs"]:
        f_raw = f_raw[f_raw["drug_name"].isin(filters["drugs"])]
    if not f_raw.empty and filters["events"]:
        f_raw = f_raw[f_raw["adverse_event"].isin(filters["events"])]
    if not f_raw.empty and filters["severity"] and "severity" in f_raw.columns:
        f_raw = f_raw[f_raw["severity"].isin(filters["severity"])]
    if not f_raw.empty and filters["country"] and "country" in f_raw.columns:
        f_raw = f_raw[f_raw["country"].isin(filters["country"])]
    if not f_raw.empty and filters["date_range"] and isinstance(filters["date_range"], tuple) and len(filters["date_range"]) == 2:
        start_date = pd.Timestamp(filters["date_range"][0])
        end_date = pd.Timestamp(filters["date_range"][1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        if "report_date" in f_raw.columns:
            f_raw = f_raw[(f_raw["report_date"] >= start_date) & (f_raw["report_date"] <= end_date)]

    if not f_scores.empty:
        if filters["drugs"]:
            f_scores = f_scores[f_scores["drug_name"].isin(filters["drugs"])]
        if filters["events"]:
            f_scores = f_scores[f_scores["adverse_event"].isin(filters["events"])]
        if filters["alert_levels"] and "alert_level" in f_scores.columns:
            f_scores = f_scores[f_scores["alert_level"].isin(filters["alert_levels"])]
        if "risk_score" in f_scores.columns:
            f_scores = f_scores[f_scores["risk_score"] >= filters["risk_threshold"]]
        if not f_raw.empty and {"drug_name", "adverse_event"}.issubset(f_raw.columns):
            valid_pairs = f_raw[["drug_name", "adverse_event"]].drop_duplicates()
            f_scores = f_scores.merge(valid_pairs, on=["drug_name", "adverse_event"], how="inner")

    return f_raw, f_scores


def choose_default_pair(scores_df: pd.DataFrame, raw_df: pd.DataFrame) -> tuple[str, str]:
    if not scores_df.empty:
        row = scores_df.sort_values("risk_score", ascending=False).iloc[0]
        return str(row["drug_name"]), str(row["adverse_event"])
    if not raw_df.empty:
        row = raw_df[["drug_name", "adverse_event"]].dropna().iloc[0]
        return str(row["drug_name"]), str(row["adverse_event"])
    return "", ""


def compute_explainability(pair_row: pd.Series) -> pd.DataFrame:
    metrics = [
        ("Risk score", float(pair_row.get("risk_score", 0)), 1.0),
        ("Signal strength", float(pair_row.get("signal_strength", 0)), 1.0),
        ("PRR", float(pair_row.get("prr", 0)), 5.0),
        ("ROR", float(pair_row.get("ror", 0)), 5.0),
        ("Drug-event ratio", float(pair_row.get("drug_event_ratio", 0)), 1.0),
        ("Co-occurrence", float(pair_row.get("co_occurrence_count", 0)), max(float(pair_row.get("drug_frequency", 1)), 1.0)),
        ("Trend slope", float(pair_row.get("time_trend_slope", 0)), 1.0),
    ]
    rows = []
    for label, value, scale in metrics:
        normalized = float(np.clip(value / scale if scale else value, 0, 1))
        rows.append({"factor": label, "value": value, "impact": round(normalized, 4)})
    exp_df = pd.DataFrame(rows).sort_values("impact", ascending=False)
    if exp_df["impact"].sum() > 0:
        exp_df["share"] = exp_df["impact"] / exp_df["impact"].sum()
    else:
        exp_df["share"] = 0.0
    return exp_df


def highlight_narrative(text: str, drug: str, event: str) -> str:
    safe = html.escape(text)
    for needle, color in [(drug, C["blue_l"]), (event, C["amber"])]:
        if needle:
            safe = safe.replace(
                html.escape(needle),
                f"<mark style='background:{color}22;color:{color};padding:0 2px;border-radius:4px'>{html.escape(needle)}</mark>",
            )
            title_needle = needle.title()
            safe = safe.replace(
                html.escape(title_needle),
                f"<mark style='background:{color}22;color:{color};padding:0 2px;border-radius:4px'>{html.escape(title_needle)}</mark>",
            )
    return safe


def build_alert_candidates(scores_df: pd.DataFrame, submissions: list, capas: list) -> list[dict]:
    alerts: list[dict] = []
    if not scores_df.empty:
        top_pairs = scores_df.nlargest(min(12, len(scores_df)), "risk_score")
        for _, row in top_pairs.iterrows():
            alert_id = f"signal::{row['drug_name']}::{row['adverse_event']}"
            alerts.append(
                {
                    "id": alert_id,
                    "source": "Signal",
                    "product_id": str(row["drug_name"]).upper().replace(" ", "-"),
                    "title": f"{str(row['drug_name']).title()} -> {str(row['adverse_event']).title()}",
                    "summary": f"Risk score {row['risk_score']:.4f} with {row.get('alert_level', 'low')} alert level.",
                    "priority": str(row.get("alert_level", "medium")).upper(),
                    "drug_name": str(row["drug_name"]),
                    "adverse_event": str(row["adverse_event"]),
                    "risk_score": float(row.get("risk_score", 0)),
                }
            )
    for sub in submissions:
        if sub.get("status", "").upper() != "SUBMITTED":
            alerts.append(
                {
                    "id": f"submission::{sub['id']}",
                    "source": "Submission",
                    "product_id": sub.get("product_id", "UNKNOWN"),
                    "title": f"{sub.get('submission_type', 'Submission')} due for {sub.get('product_id', 'Unknown')}",
                    "summary": f"Status {sub.get('status')} with risk score {sub.get('risk_score', 0):.2f}.",
                    "priority": "HIGH" if sub.get("risk_score", 0) >= 0.7 else "MEDIUM",
                    "submission_id": sub["id"],
                }
            )
    for capa in capas:
        if capa.get("state") != "CLOSED":
            alerts.append(
                {
                    "id": f"capa::{capa['id']}",
                    "source": "CAPA",
                    "product_id": capa.get("product_id", "UNKNOWN"),
                    "title": capa.get("title", f"CAPA-{capa['id']}"),
                    "summary": f"{capa.get('state', 'OPEN')} case assigned to {capa.get('assigned_to') or 'Unassigned'}.",
                    "priority": str(capa.get("priority", "MEDIUM")).upper(),
                    "capa_id": capa["id"],
                }
            )
    return alerts


def get_alert_status(alert_id: str) -> str:
    store = st.session_state.setdefault("alert_status_map", {})
    return store.get(alert_id, "NEW")


def set_alert_status(alert_id: str, status: str) -> None:
    st.session_state.setdefault("alert_status_map", {})[alert_id] = status


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
def main():
    inject_css()
    inject_scroll_observer()
    inject_nav_tab_clicks()
    bootstrap_demo_assets()

    scores = load_scores()
    raw = load_raw()
    has_s = not scores.empty
    has_r = not raw.empty

    # ── HERO ─────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            <span class="hero-title-icon">🛡️</span>
            <span class="hero-title-text">AI Pharmacovigilance Intelligence Platform</span>
        </div>
        <p class="hero-sub">
            Enterprise drug safety signal detection — analysing FDA FAERS adverse
            event data with ML & NLP to surface emerging pharmacovigilance signals
            in real time.
        </p>
        <span class="hero-badge">⚡ Signal Engine v2.0</span>
    </div>""", unsafe_allow_html=True)

    render_shell_navigation()
    filters = render_command_menu(raw, scores, has_r, has_s)
    risk_thresh = filters["risk_threshold"]

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

    # ── Apply filters ────────────────────────────────
    f_raw, f_scores = apply_global_filters(raw, scores, filters)

    # ── TABS ─────────────────────────────────────────
    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
        "🔍 Signal Detection",
        "💊 Drug Risk Analysis",
        "📈 Event Trends",
        "🌐 Network Graph",
        "🗺️ Geographic Map",
        "🧪 Risk Prediction",
        "🛡️ Audit & Compliance",
        "📜 System Activity",
    ])

    # ══════════════════════════════════════════════════
    #  TAB 1 — SIGNAL DETECTION
    # ══════════════════════════════════════════════════
    with t1:
        st.markdown('<div id="signals"></div>', unsafe_allow_html=True)
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
        st.markdown('<div id="risk"></div>', unsafe_allow_html=True)
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
        st.markdown('<div id="trends"></div>', unsafe_allow_html=True)
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
        st.markdown('<div id="network"></div>', unsafe_allow_html=True)
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
            network_html = build_network_html(f_scores, top_n=n_nodes)
        components.html(network_html, height=520, scrolling=False)

    # ══════════════════════════════════════════════════
    #  TAB 5 — GEOGRAPHIC MAP
    # ══════════════════════════════════════════════════
    with t5:
        st.markdown('<div id="map"></div>', unsafe_allow_html=True)
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
        st.markdown('<div id="predict"></div>', unsafe_allow_html=True)
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
                    result, ms, pred_err = api_predict(p_drug, p_event)

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
                        title=dict(text=f"{p_drug.title()} · {p_event.title()}", font=dict(size=12, color="#94a3b8")),
                    ))
                    st.plotly_chart(theme(fig_g, 310), use_container_width=True)

                    st.markdown(f"<p style='text-align:center;color:#55657d;font-size:0.72rem'>"
                                f"API latency: {ms:.0f}ms</p>", unsafe_allow_html=True)
                else:
                    st.error(pred_err or f"Prediction failed. Ensure the FastAPI server is reachable at {API_BASE}.")
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


    # ══════════════════════════════════════════════════
    #  TAB 7 — CAPA WORKFLOW BOARD
    # ══════════════════════════════════════════════════
    with t7:
        st.markdown('<div id="audit"></div>', unsafe_allow_html=True)
        render_capa_board()


    # ══════════════════════════════════════════════════
    #  TAB 8 — SYSTEM ACTIVITY
    # ══════════════════════════════════════════════════
    with t8:
        st.markdown('<div id="activity"></div>', unsafe_allow_html=True)
        c_head, c_btn = st.columns([5, 1])
        with c_head:
            st.markdown("### 📜 System Activity Timeline")
        with c_btn:
            if st.button("🔄 Refresh", use_container_width=True, key="refresh_system_activity"):
                fetch_events.clear()

        events = fetch_events(limit=50)
        
        if not events:
            st.info("No system events recorded yet.")
        else:
            timeline_html = '<div style="margin-top:10px; padding:0 10px;">'
            for ev in events:
                # Deterministic styling based on event type
                ev_type = ev.get('event_type', '').lower()
                if 'create' in ev_type:
                    color = C["emerald"]
                    icon = "✨"
                elif 'status' in ev_type or 'change' in ev_type:
                    color = C["blue"]
                    icon = "🔄"
                elif 'error' in ev_type or 'fail' in ev_type:
                    color = C["rose"]
                    icon = "❌"
                elif 'score' in ev_type:
                    color = C["amber"]
                    icon = "📊"
                else:
                    color = C["text_muted"]
                    icon = "📌"

                dt_fmt = pd.to_datetime(ev['created_at']).strftime("%Y-%m-%d %H:%M:%S")
                user = ev.get('user_id') or 'System'
                entity_type = str(ev.get('entity_type', '')).upper()
                desc = normalize_event_description(str(ev.get('event_description', '')))
                
                timeline_html += f"""
                <div style="border-left: 2px solid {color}; padding-left: 16px; margin-bottom: 24px; position: relative;">
                    <div style="position: absolute; left: -14px; top: 0px; background: #0b0f19; padding: 2px;">
                        {icon}
                    </div>
                    <div style="font-size: 0.75rem; color: #6b7fa0; margin-bottom: 4px; font-family: 'JetBrains Mono', monospace;">
                        {dt_fmt} &nbsp;•&nbsp; {html.escape(entity_type)} &nbsp;•&nbsp; {html.escape(str(user))}
                    </div>
                    <div style="color: {C['text']}; font-size: 0.95rem; font-weight: 500;">
                        {html.escape(desc)}
                    </div>
                </div>
                """
            timeline_html += "</div>"
            st.markdown(timeline_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
