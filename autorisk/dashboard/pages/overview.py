"""Overview page - KPIs, severity distribution, detection timeline."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from autorisk.dashboard.data_loader import SEVERITY_COLORS, SEVERITY_ORDER


def render(data: dict) -> None:
    eval_rep = data["eval_report"]
    cosmos = data["cosmos_results"]
    candidates = data["candidates"]
    gt = data["gt_labels"]

    has_eval = bool(eval_rep.get("accuracy"))

    # --- KPI Row ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clips Analyzed", len(cosmos))
    parse_ok = sum(1 for r in cosmos if r.get("parse_success"))
    k2.metric(
        "Accuracy" if has_eval else "Parse Success",
        f"{eval_rep.get('accuracy', 0):.1%}" if has_eval else f"{parse_ok}/{len(cosmos)}",
    )
    k3.metric(
        "Macro-F1" if has_eval else "MEDIUM/HIGH",
        f"{eval_rep.get('macro_f1', 0):.3f}" if has_eval
        else sum(1 for r in cosmos if r.get("severity") in ("MEDIUM", "HIGH")),
    )
    checklist = eval_rep.get("checklist_means", {}).get("mean_total", 0)
    k4.metric("Checklist", f"{checklist:.1f} / 5" if checklist else "N/A")

    st.divider()

    # --- Two column layout ---
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Severity Distribution")
        pred_counts = {s: 0 for s in SEVERITY_ORDER}
        for r in cosmos:
            sev = r.get("severity", "NONE")
            pred_counts[sev] = pred_counts.get(sev, 0) + 1

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Predicted",
            x=SEVERITY_ORDER,
            y=[pred_counts[s] for s in SEVERITY_ORDER],
            marker_color=[SEVERITY_COLORS[s] for s in SEVERITY_ORDER],
            text=[pred_counts[s] for s in SEVERITY_ORDER],
            textposition="outside",
        ))

        if gt:
            gt_counts = {s: 0 for s in SEVERITY_ORDER}
            for sev in gt.values():
                gt_counts[sev] = gt_counts.get(sev, 0) + 1
            fig.add_trace(go.Bar(
                name="Ground Truth",
                x=SEVERITY_ORDER,
                y=[gt_counts[s] for s in SEVERITY_ORDER],
                marker_color=[SEVERITY_COLORS[s] for s in SEVERITY_ORDER],
                marker_pattern_shape="/",
                text=[gt_counts[s] for s in SEVERITY_ORDER],
                textposition="outside",
                opacity=0.6,
            ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            barmode="group",
            height=350,
            margin=dict(t=30, b=40, l=40, r=20),
            legend=dict(orientation="h", y=1.12),
            yaxis_title="Count",
        )
        st.plotly_chart(fig, width="stretch")

    with col_right:
        per_class = data["analysis_report"].get("per_class_metrics", [])
        if per_class:
            st.subheader("Prediction Accuracy by Class")
            labels = [m["label"] for m in per_class]
            fig2 = go.Figure()
            for metric_name, color in [("precision", "#818CF8"), ("recall", "#34D399"), ("f1", "#FBBF24")]:
                fig2.add_trace(go.Bar(
                    name=metric_name.capitalize(),
                    x=labels,
                    y=[m[metric_name] for m in per_class],
                    text=[f"{m[metric_name]:.2f}" for m in per_class],
                    textposition="outside",
                    marker_color=color,
                ))
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                barmode="group",
                height=350,
                margin=dict(t=30, b=40, l=40, r=20),
                legend=dict(orientation="h", y=1.12),
                yaxis_range=[0, 1.15],
            )
            st.plotly_chart(fig2, width="stretch")
        else:
            st.subheader("Hazard Types")
            hazard_types: dict[str, int] = {}
            for r in cosmos:
                for h in r.get("hazards", []):
                    ht = h.get("type", "Unknown")
                    hazard_types[ht] = hazard_types.get(ht, 0) + 1
            if hazard_types:
                sorted_ht = sorted(hazard_types.items(), key=lambda x: -x[1])
                fig2 = go.Figure(go.Bar(
                    x=[h[1] for h in sorted_ht],
                    y=[h[0] for h in sorted_ht],
                    orientation="h",
                    marker_color="#818CF8",
                ))
                fig2.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350,
                    margin=dict(t=30, b=40, l=150, r=20),
                    xaxis_title="Count",
                )
                st.plotly_chart(fig2, width="stretch")
            else:
                st.info("No hazard data available yet.")

    st.divider()

    # --- Detection Timeline ---
    st.subheader("Detection Timeline")
    st.caption("Extracted danger candidates across the source video timeline, color-coded by predicted severity")

    if candidates:
        fig3 = go.Figure()
        for c in reversed(candidates):
            clip_name = Path(c["clip_path"]).name if "clip_path" in c else f"#{c['rank']}"
            sev = "NONE"
            for r in cosmos:
                if r.get("candidate_rank") == c["rank"]:
                    sev = r.get("severity", "NONE")
                    break
            gt_sev = gt.get(clip_name, "N/A") if gt else "N/A"

            fig3.add_trace(go.Bar(
                y=[f"#{c['rank']:02d}"],
                x=[c["end_sec"] - c["start_sec"]],
                base=[c["start_sec"]],
                orientation="h",
                marker_color=SEVERITY_COLORS.get(sev, "#6B7280"),
                hovertemplate=(
                    f"<b>Candidate #{c['rank']}</b><br>"
                    f"Time: {c['start_sec']:.0f}s - {c['end_sec']:.0f}s<br>"
                    f"Fused Score: {c['fused_score']:.3f}<br>"
                    f"Predicted: {sev} | GT: {gt_sev}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

        # Add legend traces
        for sev in SEVERITY_ORDER:
            fig3.add_trace(go.Bar(
                y=[None], x=[None],
                marker_color=SEVERITY_COLORS[sev],
                name=sev, showlegend=True,
            ))

        fig3.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=max(350, len(candidates) * 22 + 80),
            margin=dict(t=10, b=40, l=60, r=20),
            xaxis_title="Time in Source Video (seconds)",
            legend=dict(orientation="h", y=1.05),
            bargap=0.3,
        )
        st.plotly_chart(fig3, width="stretch")

    st.divider()

    # --- Pipeline Architecture ---
    st.subheader("Pipeline Architecture")
    _render_pipeline_diagram()


def _render_pipeline_diagram() -> None:
    """Render pipeline architecture as styled HTML."""
    html = """
    <div style="display:flex; align-items:center; justify-content:center; gap:8px; flex-wrap:wrap; padding:20px 0;">
        <div style="background:linear-gradient(135deg,#1E3A5F,#2563EB); border-radius:12px; padding:16px 20px; text-align:center; min-width:140px;">
            <div style="font-size:24px;">&#127909;</div>
            <div style="font-weight:700; font-size:14px; color:#BFDBFE;">B1: Mining</div>
            <div style="font-size:11px; color:#93C5FD; margin-top:4px;">Audio + Motion + Proximity</div>
        </div>
        <div style="font-size:24px; color:#6B7280;">&#10132;</div>
        <div style="background:linear-gradient(135deg,#3B1F6E,#7C3AED); border-radius:12px; padding:16px 20px; text-align:center; min-width:140px;">
            <div style="font-size:24px;">&#129302;</div>
            <div style="font-weight:700; font-size:14px; color:#DDD6FE;">B2: Cosmos VLM</div>
            <div style="font-size:11px; color:#C4B5FD; margin-top:4px;">Causal Risk Assessment</div>
        </div>
        <div style="font-size:24px; color:#6B7280;">&#10132;</div>
        <div style="background:linear-gradient(135deg,#713F12,#D97706); border-radius:12px; padding:16px 20px; text-align:center; min-width:140px;">
            <div style="font-size:24px;">&#128202;</div>
            <div style="font-weight:700; font-size:14px; color:#FEF3C7;">B3: Ranking</div>
            <div style="font-size:11px; color:#FDE68A; margin-top:4px;">Severity + Tiebreakers</div>
        </div>
        <div style="font-size:24px; color:#6B7280;">&#10132;</div>
        <div style="background:linear-gradient(135deg,#064E3B,#059669); border-radius:12px; padding:16px 20px; text-align:center; min-width:140px;">
            <div style="font-size:24px;">&#9989;</div>
            <div style="font-weight:700; font-size:14px; color:#D1FAE5;">B4: Evaluation</div>
            <div style="font-size:11px; color:#A7F3D0; margin-top:4px;">Metrics + Checklist</div>
        </div>
        <div style="font-size:24px; color:#6B7280;">&#10132;</div>
        <div style="background:linear-gradient(135deg,#7F1D1D,#DC2626); border-radius:12px; padding:16px 20px; text-align:center; min-width:140px;">
            <div style="font-size:24px;">&#128300;</div>
            <div style="font-weight:700; font-size:14px; color:#FEE2E2;">B5: Analysis</div>
            <div style="font-size:11px; color:#FECACA; margin-top:4px;">TTC + Grounding + Saliency</div>
        </div>
        <div style="font-size:24px; color:#6B7280;">&#10132;</div>
        <div style="background:linear-gradient(135deg,#0C4A6E,#0EA5E9); border-radius:12px; padding:16px 20px; text-align:center; min-width:140px;">
            <div style="font-size:24px;">&#127902;</div>
            <div style="font-weight:700; font-size:14px; color:#E0F2FE;">Predict 2</div>
            <div style="font-size:11px; color:#BAE6FD; margin-top:4px;">Future Video Generation</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# Need Path for clip_path parsing
from pathlib import Path  # noqa: E402
