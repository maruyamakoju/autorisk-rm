"""AutoRisk-RM Interactive Dashboard.

Launch with:
    streamlit run autorisk/dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from autorisk.dashboard.data_loader import load_data
from autorisk.dashboard.pages import comparison, depth, evaluation, explorer, overview, search, signals

# --- Page Config ---
st.set_page_config(
    page_title="AutoRisk-RM Dashboard",
    page_icon="https://developer.nvidia.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Tighten padding */
    .block-container { padding-top: 1.5rem; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1E1E2E 0%, #2D2D44 100%);
        border: 1px solid rgba(124, 58, 237, 0.15);
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }

    /* Divider */
    hr { border-color: rgba(124, 58, 237, 0.15) !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1A1A2E 100%);
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown(
        '<h2 style="text-align:center; margin-bottom:0;">AutoRisk-RM</h2>'
        '<p style="text-align:center; color:#9CA3AF; font-size:13px; margin-top:4px;">'
        'Dashcam Danger Extraction Pipeline</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Run directory selector - auto-detect available runs
    outputs_root = Path(__file__).resolve().parent.parent.parent / "outputs"
    available_runs = {}
    run_labels = {
        "public_run": "UK Urban (Daytime)",
        "japan_run": "Japan (Urban)",
        "winter_run": "Winter / Snow",
        "us_highway_run": "US Highway",
    }
    if outputs_root.exists():
        for d in sorted(outputs_root.iterdir()):
            if d.is_dir() and (d / "cosmos_results.json").exists():
                label = run_labels.get(d.name, d.name)
                available_runs[label] = str(d)

    if available_runs:
        selected_label = st.selectbox(
            "Select Run",
            list(available_runs.keys()),
            index=0,
        )
        run_dir = available_runs[selected_label]
    else:
        default_run = str(outputs_root / "public_run")
        run_dir = st.text_input("Run Directory", value=default_run)

    st.divider()

    # Navigation
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Clip Explorer",
            "Search",
            "Evaluation",
            "Signal Analysis",
            "Technical Depth",
            "Cross-Run Comparison",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Powered by NVIDIA Cosmos Reason 2")
    st.caption("Built with Streamlit + Plotly")

# --- Load Data ---
if not Path(run_dir).exists():
    st.error(f"Run directory not found: {run_dir}")
    st.stop()

data = load_data(run_dir)

# --- Header ---
st.markdown(
    '<h1 style="margin-bottom:0;">AutoRisk-RM</h1>'
    '<p style="color:#9CA3AF; margin-top:0;">End-to-End Dashcam Danger Extraction '
    '&amp; Causal Risk Assessment</p>',
    unsafe_allow_html=True,
)

warnings = list(data.get("load_warnings", []))
if warnings:
    st.warning(
        f"Data load completed with {len(warnings)} warning(s). Some dashboard sections may be partial."
    )
    with st.expander("Show data load warnings"):
        for msg in warnings[:100]:
            st.text(f"- {msg}")
        if len(warnings) > 100:
            st.text(f"... ({len(warnings) - 100} more)")

# --- Page Routing ---
if page == "Overview":
    overview.render(data)
elif page == "Clip Explorer":
    explorer.render(data)
elif page == "Search":
    search.render(data)
elif page == "Evaluation":
    evaluation.render(data)
elif page == "Signal Analysis":
    signals.render(data)
elif page == "Technical Depth":
    depth.render(data)
elif page == "Cross-Run Comparison":
    comparison.render(data)
