"""Technical Depth page - TTC, Grounding, Calibration, Saliency analysis."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from autorisk.dashboard.data_loader import SEVERITY_COLORS, SEVERITY_ORDER


def render(data: dict) -> None:
    tab_ttc, tab_ground, tab_calib, tab_sal = st.tabs([
        "Time-to-Collision", "Cross-Modal Grounding",
        "Confidence Calibration", "Gradient Saliency",
    ])

    with tab_ttc:
        _render_ttc(data)

    with tab_ground:
        _render_grounding(data)

    with tab_calib:
        _render_calibration(data)

    with tab_sal:
        _render_saliency(data)


def _render_ttc(data: dict) -> None:
    ttc_results = data["ttc_results"]
    gt = data["gt_labels"]
    cosmos = data["cosmos_results"]

    if not ttc_results:
        st.warning("No TTC results found.")
        return

    st.markdown(
        "**Time-to-Collision (TTC)** measures how many seconds until a potential collision "
        "based on object tracking and bounding box expansion rates."
    )

    # Build scatter data
    scatter_data = []
    for t in ttc_results:
        clip_name = Path(t["clip_path"]).name
        gt_sev = gt.get(clip_name, "NONE")
        min_ttc = t.get("min_ttc", -1)
        if min_ttc > 0:
            scatter_data.append({
                "clip": clip_name,
                "min_ttc": min_ttc,
                "severity": gt_sev,
                "n_tracks": t.get("n_tracks", 0),
                "n_critical": t.get("n_critical", 0),
            })

    if scatter_data:
        # KPI row
        k1, k2, k3 = st.columns(3)
        all_ttc = [d["min_ttc"] for d in scatter_data]
        k1.metric("Mean Min-TTC", f"{np.mean(all_ttc):.2f}s")
        k2.metric("Lowest TTC", f"{min(all_ttc):.2f}s")
        k3.metric("Clips with TTC < 1.5s", sum(1 for t in all_ttc if t < 1.5))

        st.divider()

        # Scatter: min_TTC by severity
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Min TTC by Severity Class**")
            fig = go.Figure()
            for sev in SEVERITY_ORDER:
                pts = [d for d in scatter_data if d["severity"] == sev]
                if pts:
                    fig.add_trace(go.Box(
                        y=[p["min_ttc"] for p in pts],
                        name=sev,
                        marker_color=SEVERITY_COLORS[sev],
                        boxpoints="all",
                        jitter=0.3,
                        pointpos=-1.8,
                    ))
            fig.add_hline(y=1.5, line_dash="dash", line_color="#EF4444",
                          annotation_text="Critical Threshold")
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(t=20, b=40, l=40, r=20),
                yaxis_title="Min TTC (seconds)",
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.markdown("**TTC vs Fused Score**")
            fig2 = go.Figure()
            for sev in SEVERITY_ORDER:
                pts = [d for d in scatter_data if d["severity"] == sev]
                if pts:
                    # Find fused scores
                    for p in pts:
                        for r in cosmos:
                            if Path(r["clip_path"]).name == p["clip"]:
                                p["fused_score"] = r.get("fused_score", 0)
                    fig2.add_trace(go.Scatter(
                        x=[p.get("fused_score", 0) for p in pts],
                        y=[p["min_ttc"] for p in pts],
                        mode="markers",
                        name=sev,
                        marker=dict(color=SEVERITY_COLORS[sev], size=10),
                        text=[p["clip"] for p in pts],
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "Fused Score: %{x:.3f}<br>"
                            "Min TTC: %{y:.2f}s<extra></extra>"
                        ),
                    ))
            fig2.add_hline(y=1.5, line_dash="dash", line_color="#EF4444")
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(t=20, b=40, l=40, r=20),
                xaxis_title="Fused Mining Score",
                yaxis_title="Min TTC (seconds)",
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig2, width="stretch")


def _render_grounding(data: dict) -> None:
    report = data["grounding_report"]
    if not report:
        st.warning("No grounding report found.")
        return

    st.markdown(
        "**Cross-Modal Grounding** measures whether the VLM's textual output is "
        "consistent with the mining signals that triggered candidate extraction."
    )

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mean Grounding", f"{report.get('mean_grounding_score', 0):.0%}")
    k2.metric("Fully Grounded", f"{report.get('n_fully_grounded', 0)} / {report.get('n_clips', 0)}")
    k3.metric("Hallucinations", report.get("n_has_hallucination", 0))
    k4.metric("Ungrounded", report.get("n_has_ungrounded", 0))

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        # Signal grounding rates - radar chart
        st.markdown("**Per-Signal Grounding Rate**")
        rates = report.get("signal_grounding_rates", {})
        if rates:
            cats = [k.capitalize() for k in rates.keys()]
            vals = list(rates.values())
            cats_closed = cats + [cats[0]]
            vals_closed = vals + [vals[0]]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=cats_closed,
                fill="toself",
                fillcolor="rgba(16, 185, 129, 0.2)",
                line=dict(color="#10B981", width=2),
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(range=[0, 1], showticklabels=True, gridcolor="#333"),
                    angularaxis=dict(gridcolor="#333"),
                ),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(t=30, b=30, l=60, r=60),
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")

    with col2:
        # Grounding by severity
        st.markdown("**Grounding by Severity**")
        by_sev = report.get("grounding_by_severity", {})
        if by_sev:
            fig2 = go.Figure()
            sevs = [s for s in SEVERITY_ORDER if s in by_sev]
            fig2.add_trace(go.Bar(
                x=sevs,
                y=[by_sev[s] for s in sevs],
                marker_color=[SEVERITY_COLORS[s] for s in sevs],
                text=[f"{by_sev[s]:.0%}" for s in sevs],
                textposition="outside",
            ))
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(t=20, b=40, l=40, r=20),
                yaxis_range=[0, 1.15],
                yaxis_title="Grounding Rate",
                showlegend=False,
            )
            st.plotly_chart(fig2, width="stretch")

    st.markdown(
        "> **Insight:** Audio grounding is low (25%) because the VLM processes video only "
        "and cannot hear audio signals. This validates that Cosmos relies on visual evidence "
        "rather than hallucinating audio cues."
    )


def _render_calibration(data: dict) -> None:
    report = data["calibration_report"]
    if not report:
        st.warning("No calibration report found.")
        return

    st.markdown(
        "**Confidence Calibration** assesses whether the model's stated confidence "
        "aligns with its actual accuracy. Temperature scaling is applied post-hoc."
    )

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ECE (Before)", f"{report.get('ece', 0):.4f}")
    k2.metric("ECE (After)", f"{report.get('ece_after', 0):.4f}",
              delta=f"{report.get('ece_after', 0) - report.get('ece', 0):.4f}")
    k3.metric("Temperature", f"{report.get('optimal_temperature', 1):.1f}")
    improvement = (1 - report.get("ece_after", 0) / max(report.get("ece", 0.001), 0.001)) * 100
    k4.metric("Improvement", f"{improvement:.0f}%")

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        # Reliability diagram (before)
        st.markdown("**Reliability Diagram (Before T-Scaling)**")
        bins = report.get("bins", [])
        _plot_reliability(bins, "Before")

    with col2:
        # Reliability diagram (after)
        st.markdown("**Reliability Diagram (After T-Scaling)**")
        bins_after = report.get("bins_after", [])
        _plot_reliability(bins_after, "After")

    st.divider()

    # Per-severity confidence
    st.subheader("Per-Severity Confidence Analysis")
    conf_by_sev = report.get("confidence_by_severity", {})
    if conf_by_sev:
        cols = st.columns(4)
        for i, sev in enumerate(SEVERITY_ORDER):
            if sev in conf_by_sev:
                info = conf_by_sev[sev]
                with cols[i]:
                    st.markdown(
                        f'<div style="border-left:4px solid {SEVERITY_COLORS[sev]}; '
                        f'padding-left:12px;">'
                        f'<b>{sev}</b> (n={info["n_samples"]})<br>'
                        f'Confidence: {info["mean_confidence"]:.2f}<br>'
                        f'Accuracy: {info["accuracy"]:.2f}<br>'
                        f'{"Overconfident" if info.get("overconfident") else "Underconfident"}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


def _plot_reliability(bins: list[dict], label: str) -> None:
    if not bins:
        st.caption("No bin data available.")
        return

    non_empty = [b for b in bins if b.get("count", 0) > 0]
    if not non_empty:
        st.caption("No non-empty bins.")
        return

    fig = go.Figure()
    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="#6B7280", dash="dash"),
        name="Perfect",
        showlegend=True,
    ))
    # Actual bars
    mid_points = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in non_empty]
    fig.add_trace(go.Bar(
        x=mid_points,
        y=[b["avg_accuracy"] for b in non_empty],
        width=0.08,
        marker_color="#7C3AED",
        name="Accuracy",
        text=[f"n={b['count']}" for b in non_empty],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(t=20, b=40, l=40, r=20),
        xaxis_title="Confidence",
        yaxis_title="Accuracy",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.15],
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, width="stretch")


def _render_saliency(data: dict) -> None:
    saliency_results = data["saliency_results"]
    saliency_images = data["saliency_images"]
    gt = data["gt_labels"]
    cosmos = data["cosmos_results"]

    if not saliency_results:
        st.warning("No saliency results found.")
        return

    st.markdown(
        "**Gradient Saliency** computes the gradient of the model's next-token logit "
        "w.r.t. visual input, revealing which spatial regions most influence risk assessment."
    )

    st.markdown(
        f"Processed **{len(saliency_results)}** clips | "
        f"Spatial grid: **{saliency_results[0].get('spatial_grid', [0, 0])[0]}x"
        f"{saliency_results[0].get('spatial_grid', [0, 0])[1]}** | "
        f"Temporal frames: **{saliency_results[0].get('n_temporal_frames', 0)}**"
    )

    st.divider()

    # Gallery
    cols_per_row = 2
    for i in range(0, len(saliency_results), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx >= len(saliency_results):
                break
            sal = saliency_results[idx]
            clip_name = sal["clip_name"]
            images = saliency_images.get(clip_name, {})

            # Find severity
            sev = "?"
            for r in cosmos:
                if Path(r["clip_path"]).name == clip_name:
                    sev = r.get("severity", "?")
                    break
            gt_sev = gt.get(clip_name, "?")

            with cols[j]:
                st.markdown(
                    f"**{clip_name}** &nbsp; "
                    f"Pred: {sev} | GT: {gt_sev} &nbsp; "
                    f"Peak frame: {sal['peak_frame_idx']}/{sal['n_temporal_frames']}"
                )
                if images.get("heatmap_b64"):
                    sub1, sub2 = st.columns(2)
                    with sub1:
                        raw_bytes = base64.b64decode(images["raw_frame_b64"])
                        st.image(raw_bytes, caption="Raw", width="stretch")
                    with sub2:
                        heat_bytes = base64.b64decode(images["heatmap_b64"])
                        st.image(heat_bytes, caption="Saliency", width="stretch")
                else:
                    st.caption("No heatmap available")

    st.divider()

    # Temporal saliency chart
    st.subheader("Temporal Attention Distribution")
    fig = go.Figure()
    for sal in saliency_results:
        sums = sal.get("frame_saliency_sums", [])
        fig.add_trace(go.Scatter(
            x=list(range(len(sums))),
            y=sums,
            mode="lines+markers",
            name=sal["clip_name"].replace("candidate_", "#").replace(".mp4", ""),
            marker=dict(size=5),
        ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(t=20, b=40, l=40, r=20),
        xaxis_title="Temporal Frame Index",
        yaxis_title="Saliency Sum",
        legend=dict(orientation="h", y=1.15, font=dict(size=10)),
    )
    st.plotly_chart(fig, width="stretch")
