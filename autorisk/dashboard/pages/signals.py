"""Signals page - signal contribution analysis, correlation, ablation."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from autorisk.dashboard.data_loader import SEVERITY_COLORS, SEVERITY_ORDER


def render(data: dict) -> None:
    analysis = data["analysis_report"]
    ablation = data["ablation_results"]

    # --- Signal-Severity Heatmap ---
    st.subheader("Signal-Severity Heatmap")
    st.caption("Mean signal activation score by ground-truth severity class")

    heatmap_data = analysis.get("signal_heatmap", {})
    if heatmap_data:
        signals = list(heatmap_data.keys())
        matrix = []
        for sig in signals:
            row = [heatmap_data[sig].get(sev, 0) for sev in SEVERITY_ORDER]
            matrix.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=SEVERITY_ORDER,
            y=[s.capitalize() for s in signals],
            colorscale="Viridis",
            text=[[f"{v:.3f}" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont=dict(size=14),
            hovertemplate=(
                "Signal: %{y}<br>"
                "Severity: %{x}<br>"
                "Mean Score: %{z:.3f}<extra></extra>"
            ),
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=250,
            margin=dict(t=20, b=40, l=80, r=20),
        )
        st.plotly_chart(fig, width="stretch")

    st.divider()

    # --- Correlation Table ---
    col_corr, col_thresh = st.columns([1, 1])

    with col_corr:
        st.subheader("Spearman Correlation with Severity")
        signal_analysis = analysis.get("signal_analysis", [])
        if signal_analysis:
            for sa in signal_analysis:
                name = sa["signal_name"].capitalize()
                rho = sa["spearman_rho"]
                p = sa["spearman_p"]
                sig_marker = "**" if p < 0.05 else ""

                bar_width = abs(rho) * 200
                bar_color = "#10B981" if rho > 0 else "#EF4444"
                st.markdown(
                    f"**{name}**: {sig_marker}rho = {rho:+.3f}{sig_marker} "
                    f"(p = {p:.3f})"
                )

            # Correlation bar chart
            fig2 = go.Figure()
            names = [sa["signal_name"].capitalize() for sa in signal_analysis]
            rhos = [sa["spearman_rho"] for sa in signal_analysis]
            ps = [sa["spearman_p"] for sa in signal_analysis]
            colors = ["#10B981" if p < 0.05 else "#6B7280" for p in ps]

            fig2.add_trace(go.Bar(
                x=names,
                y=rhos,
                marker_color=colors,
                text=[f"{r:+.3f}" for r in rhos],
                textposition="outside",
            ))
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(t=20, b=40, l=40, r=20),
                yaxis_title="Spearman rho",
                yaxis_range=[-0.5, 0.5],
            )
            st.plotly_chart(fig2, width="stretch")

    with col_thresh:
        st.subheader("Signal Threshold Performance")
        if signal_analysis:
            fig3 = go.Figure()
            names = [sa["signal_name"].capitalize() for sa in signal_analysis]
            accs = [sa["threshold_accuracy"] for sa in signal_analysis]
            f1s = [sa["threshold_f1"] for sa in signal_analysis]

            fig3.add_trace(go.Bar(
                name="Accuracy",
                x=names, y=accs,
                marker_color="#818CF8",
                text=[f"{v:.2f}" for v in accs],
                textposition="outside",
            ))
            fig3.add_trace(go.Bar(
                name="F1",
                x=names, y=f1s,
                marker_color="#34D399",
                text=[f"{v:.2f}" for v in f1s],
                textposition="outside",
            ))
            fig3.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                barmode="group",
                height=300,
                margin=dict(t=20, b=40, l=40, r=20),
                yaxis_range=[0, 0.6],
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig3, width="stretch")

    st.divider()

    # --- Ablation Study ---
    st.subheader("Ablation Study")
    st.caption("Comparing pipeline with and without Cosmos Reason 2 VLM inference")

    if ablation:
        cols = st.columns(len(ablation))
        for i, ab in enumerate(ablation):
            with cols[i]:
                mode = ab.get("mode", "unknown")
                desc = ab.get("description", "")
                is_full = mode == "cosmos_video"

                border_color = "#7C3AED" if is_full else "#374151"
                st.markdown(
                    f'<div style="border:2px solid {border_color}; border-radius:12px; '
                    f'padding:20px; text-align:center;">'
                    f'<h4 style="margin:0 0 8px 0;">{mode.replace("_", " ").title()}</h4>'
                    f'<p style="color:#9CA3AF; font-size:12px; margin-bottom:16px;">{desc}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.metric("Accuracy", f"{ab.get('accuracy', 0):.1%}")
                st.metric("Macro-F1", f"{ab.get('macro_f1', 0):.3f}")
                st.metric("Checklist", f"{ab.get('checklist_mean', 0):.1f} / 5")

        # Improvement summary
        if len(ablation) >= 2:
            base = ablation[0]
            full = ablation[1]
            f1_delta = full.get("macro_f1", 0) - base.get("macro_f1", 0)
            cl_delta = full.get("checklist_mean", 0) - base.get("checklist_mean", 0)
            st.markdown(
                f"**Cosmos VLM Impact:** F1 {f1_delta:+.3f}, "
                f"Checklist {cl_delta:+.1f}"
            )
