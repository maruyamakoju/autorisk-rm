"""Cross-run comparison page - compare results across multiple video sources."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from autorisk.dashboard.comparison_utils import (
    SEVERITY_COLORS,
    SEVERITY_ORDER,
    SIGNAL_NAMES,
    compute_run_kpis,
    confidence_values,
    discover_runs,
    mean_signal_scores,
    positive_min_ttc_values,
    severity_counts,
    severity_ratios,
)


def render(data: dict) -> None:  # noqa: ARG001
    """Render cross-run comparison."""
    outputs_root = Path(__file__).resolve().parent.parent.parent.parent / "outputs"

    # Discover all completed runs.
    runs, skipped = discover_runs(outputs_root=outputs_root)

    if skipped:
        st.warning(
            f"Skipped {len(skipped)} run(s) due to load errors. Comparison uses only successfully loaded runs."
        )
        with st.expander("Show skipped runs"):
            for msg in skipped:
                st.text(f"- {msg}")

    if len(runs) < 2:
        st.info(
            f"Cross-run comparison requires at least 2 completed runs. "
            f"Found {len(runs)}. Run the pipeline on additional videos first."
        )
        return

    st.caption(f"Comparing {len(runs)} pipeline runs across different video sources")

    # --- Run Summary Cards ---
    st.subheader("Run Summaries")
    _render_run_summary_cards(runs)

    st.divider()

    # --- KPI Comparison Table ---
    st.subheader("Performance Comparison")
    cols = st.columns(len(runs))
    for i, (label, rdata) in enumerate(runs.items()):
        kpis = compute_run_kpis(rdata)
        with cols[i]:
            st.markdown(f"**{label}**")
            n_clips = int(kpis["clips"])
            parse_ok = int(kpis["parse_ok"])
            acc = float(kpis["accuracy"])
            f1 = float(kpis["macro_f1"])
            checklist = float(kpis["checklist_mean_total"])

            st.metric("Clips", n_clips)
            st.metric("Parse OK", f"{parse_ok}/{n_clips}")
            st.metric("Accuracy", f"{acc:.1%}" if acc else "N/A")
            st.metric("F1", f"{f1:.3f}" if f1 else "N/A")
            st.metric("Checklist", f"{checklist:.1f}/5" if checklist else "N/A")

    st.divider()

    # --- Severity Distribution Comparison ---
    st.subheader("Severity Distribution by Source")
    fig = go.Figure()
    for label, rdata in runs.items():
        cosmos = rdata.get("cosmos_results", [])
        cosmos_list = cosmos if isinstance(cosmos, list) else []
        counts = severity_counts(cosmos_list)
        ratios = severity_ratios(cosmos_list)
        fig.add_trace(go.Bar(
            name=label,
            x=SEVERITY_ORDER,
            y=[ratios[s] for s in SEVERITY_ORDER],
            text=[f"{counts[s]}" for s in SEVERITY_ORDER],
            textposition="outside",
        ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="group",
        height=350,
        margin=dict(t=30, b=40, l=40, r=20),
        yaxis_title="Proportion",
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, width="stretch")

    st.divider()

    # --- Confidence Distribution ---
    st.subheader("Confidence Distribution by Source")
    fig2 = go.Figure()
    for label, rdata in runs.items():
        cosmos = rdata.get("cosmos_results", [])
        cosmos_list = cosmos if isinstance(cosmos, list) else []
        confs = confidence_values(cosmos_list)
        if confs:
            fig2.add_trace(go.Box(
                y=confs,
                name=label,
                boxpoints="all",
                jitter=0.3,
            ))
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(t=20, b=40, l=40, r=20),
        yaxis_title="Confidence",
        showlegend=False,
    )
    st.plotly_chart(fig2, width="stretch")

    st.divider()

    # --- Mining Signal Comparison ---
    st.subheader("Mining Signal Scores by Source")
    fig3 = go.Figure()
    for label, rdata in runs.items():
        candidates = rdata.get("candidates", [])
        candidates_list = candidates if isinstance(candidates, list) else []
        if not candidates_list:
            continue
        means = mean_signal_scores(candidates_list)
        fig3.add_trace(go.Bar(
            name=label,
            x=[s.capitalize() for s in SIGNAL_NAMES],
            y=[means[s] for s in SIGNAL_NAMES],
            text=[f"{means[s]:.2f}" for s in SIGNAL_NAMES],
            textposition="outside",
        ))
    fig3.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="group",
        height=300,
        margin=dict(t=20, b=40, l=40, r=20),
        yaxis_title="Mean Score",
        yaxis_range=[0, 1],
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig3, width="stretch")

    st.divider()

    # --- TTC Comparison ---
    st.subheader("Time-to-Collision by Source")
    fig4 = go.Figure()
    for label, rdata in runs.items():
        ttc = rdata.get("ttc_results", [])
        ttc_list = ttc if isinstance(ttc, list) else []
        if not ttc_list:
            continue
        min_ttcs = positive_min_ttc_values(ttc_list)
        if min_ttcs:
            fig4.add_trace(go.Box(
                y=min_ttcs,
                name=label,
                boxpoints="all",
                jitter=0.3,
            ))
    fig4.add_hline(y=1.5, line_dash="dash", line_color="#EF4444",
                   annotation_text="Critical (1.5s)")
    fig4.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(t=20, b=40, l=40, r=20),
        yaxis_title="Min TTC (seconds)",
        showlegend=False,
    )
    st.plotly_chart(fig4, width="stretch")

    # --- Grounding Comparison ---
    grounding_data = {}
    for label, rdata in runs.items():
        gr = rdata.get("grounding_report", {})
        if gr:
            grounding_data[label] = gr.get("mean_grounding_score", 0)

    if grounding_data:
        st.divider()
        st.subheader("Cross-Modal Grounding by Source")
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            x=list(grounding_data.keys()),
            y=list(grounding_data.values()),
            text=[f"{v:.0%}" for v in grounding_data.values()],
            textposition="outside",
            marker_color="#10B981",
        ))
        fig5.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(t=20, b=40, l=40, r=20),
            yaxis_title="Mean Grounding Score",
            yaxis_range=[0, 1.15],
            showlegend=False,
        )
        st.plotly_chart(fig5, width="stretch")


def _render_run_summary_cards(runs: dict) -> None:
    """Render summary cards with headline stats for each run."""
    cols = st.columns(len(runs))
    for i, (label, rdata) in enumerate(runs.items()):
        with cols[i]:
            cosmos = rdata.get("cosmos_results", [])
            cosmos_list = cosmos if isinstance(cosmos, list) else []
            kpis = compute_run_kpis(rdata)

            # Count severity distribution
            counts = severity_counts(cosmos_list)
            n_clips = int(kpis["clips"])
            parse_rate = (int(kpis["parse_ok"]) / n_clips * 100) if n_clips > 0 else 0

            # Build colored severity bars
            sev_bars = ""
            for sev in SEVERITY_ORDER:
                count = counts[sev]
                if count > 0:
                    color = SEVERITY_COLORS.get(sev, "#6B7280")
                    sev_bars += f'<div style="background:{color}; color:white; padding:2px 6px; border-radius:4px; font-size:10px; font-weight:700; margin-right:4px;">{sev}: {count}</div>'

            html = f"""
            <div style="background:linear-gradient(135deg, #1E1E2E 0%, #2D2D44 100%);
                        border:1px solid rgba(124, 58, 237, 0.15);
                        border-radius:10px;
                        padding:14px;">
                <div style="font-size:16px; font-weight:700; margin-bottom:8px; color:#A78BFA;">{label}</div>
                <div style="font-size:13px; color:#9CA3AF; margin-bottom:6px;">
                    <strong>{n_clips}</strong> clips analyzed
                </div>
                <div style="font-size:13px; color:#9CA3AF; margin-bottom:8px;">
                    Parse: <strong>{parse_rate:.0f}%</strong>
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:4px;">
                    {sev_bars}
                </div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
