"""Evaluation page - confusion matrix, per-class metrics, error analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from autorisk.dashboard.data_loader import SEVERITY_COLORS, SEVERITY_ORDER


def render(data: dict) -> None:
    eval_rep = data["eval_report"]
    analysis = data["analysis_report"]

    if not eval_rep:
        st.warning("No evaluation report found.")
        return

    # --- KPI Row ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{eval_rep.get('accuracy', 0):.1%}")
    k2.metric("Macro-F1", f"{eval_rep.get('macro_f1', 0):.3f}")
    checklist = eval_rep.get("checklist_means", {}).get("mean_total", 0)
    k3.metric("Checklist", f"{checklist:.1f} / 5")
    k4.metric("Errors", f"{eval_rep.get('n_failures', 0)} / {eval_rep.get('n_samples', 0)}")

    st.divider()

    # --- Confusion Matrix + Error Breakdown ---
    col_cm, col_err = st.columns([1, 1])

    with col_cm:
        st.subheader("Confusion Matrix")
        cm = eval_rep.get("confusion_matrix", {})
        if cm:
            matrix = []
            for actual in SEVERITY_ORDER:
                row = []
                for predicted in SEVERITY_ORDER:
                    row.append(cm.get(actual, {}).get(predicted, 0))
                matrix.append(row)

            matrix_np = np.array(matrix)

            # Annotated heatmap
            fig = go.Figure(data=go.Heatmap(
                z=matrix_np,
                x=SEVERITY_ORDER,
                y=SEVERITY_ORDER,
                colorscale=[
                    [0, "#1E1E2E"],
                    [0.25, "#312E81"],
                    [0.5, "#5B21B6"],
                    [0.75, "#7C3AED"],
                    [1.0, "#A78BFA"],
                ],
                text=matrix_np,
                texttemplate="%{text}",
                textfont=dict(size=18, color="white"),
                hovertemplate=(
                    "Actual: %{y}<br>"
                    "Predicted: %{x}<br>"
                    "Count: %{z}<extra></extra>"
                ),
                showscale=False,
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Predicted",
                yaxis_title="Actual (Ground Truth)",
                yaxis=dict(autorange="reversed"),
                height=380,
                margin=dict(t=20, b=60, l=80, r=20),
            )
            st.plotly_chart(fig, width="stretch")

    with col_err:
        st.subheader("Error Analysis")
        err = analysis.get("error_summary", {})
        if err:
            total = err.get("total_errors", 0)
            over = err.get("over_estimation", 0)
            under = err.get("under_estimation", 0)
            adjacent = err.get("adjacent_miss", 0)
            major = err.get("major_miss", 0)

            # Donut chart
            fig2 = go.Figure()
            fig2.add_trace(go.Pie(
                labels=["Over-estimation", "Under-estimation"],
                values=[over, under],
                hole=0.5,
                marker_colors=["#F59E0B", "#3B82F6"],
                textinfo="label+value",
                textfont=dict(size=13),
            ))
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False,
                annotations=[dict(
                    text=f"<b>{total}</b><br>errors",
                    x=0.5, y=0.5, font_size=16, showarrow=False,
                    font_color="#FAFAFA",
                )],
            )
            st.plotly_chart(fig2, width="stretch")

            # Summary stats
            st.markdown(
                f"- **Adjacent miss** (off by 1 level): {adjacent} ({adjacent / max(total, 1):.0%})\n"
                f"- **Major miss** (off by 2+ levels): {major} ({major / max(total, 1):.0%})\n"
                f"- **Over-estimation bias**: {over / max(total, 1):.0%} "
                f"(conservative safety bias)"
            )

    st.divider()

    # --- Checklist Breakdown ---
    st.subheader("Explanation Checklist")
    checklist_data = eval_rep.get("checklist_means", {})
    items = [
        ("actors_accurate", "Actors Accurate"),
        ("causal_clear", "Causal Clear"),
        ("spatial_specific", "Spatial Specific"),
        ("prediction_plausible", "Prediction Plausible"),
        ("action_reasonable", "Action Reasonable"),
    ]
    cols = st.columns(5)
    for i, (key, label) in enumerate(items):
        val = checklist_data.get(key, 0)
        cols[i].metric(label, f"{val:.2f}")

    st.divider()

    # --- Enhanced Correction Before/After ---
    correction_report = data.get("correction_report")
    if correction_report:
        st.subheader("Signal-Based Correction (Before / After)")

        before_acc = eval_rep.get("accuracy", 0)
        after_acc = correction_report.get("accuracy", 0)
        before_f1 = eval_rep.get("macro_f1", 0)
        after_f1 = correction_report.get("macro_f1", 0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy (Before)", f"{before_acc:.1%}")
        c2.metric("Accuracy (After)", f"{after_acc:.1%}", delta=f"+{after_acc - before_acc:.1%}")
        c3.metric("Macro-F1 (Before)", f"{before_f1:.3f}")
        c4.metric("Macro-F1 (After)", f"{after_f1:.3f}", delta=f"+{after_f1 - before_f1:.3f}")

        # Per-class recall comparison
        corr_per_class = correction_report.get("per_class", {})
        orig_per_class = analysis.get("per_class_metrics", [])
        if corr_per_class:
            fig_corr = go.Figure()
            sev_labels = list(corr_per_class.keys())

            # Before recall (from original eval)
            before_recalls = []
            for sev in sev_labels:
                orig = next((m for m in orig_per_class if m.get("label") == sev), None)
                before_recalls.append(orig["recall"] if orig else 0)

            after_recalls = [corr_per_class[s].get("recall", 0) for s in sev_labels]

            fig_corr.add_trace(go.Bar(
                name="Before Correction",
                x=sev_labels,
                y=before_recalls,
                marker_color="#6B7280",
                text=[f"{v:.0%}" for v in before_recalls],
                textposition="outside",
            ))
            fig_corr.add_trace(go.Bar(
                name="After Correction",
                x=sev_labels,
                y=after_recalls,
                marker_color="#10B981",
                text=[f"{v:.0%}" for v in after_recalls],
                textposition="outside",
            ))
            fig_corr.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                barmode="group",
                height=300,
                margin=dict(t=30, b=40, l=40, r=20),
                legend=dict(orientation="h", y=1.12),
                yaxis_title="Recall",
                yaxis_range=[0, 1.15],
            )
            st.plotly_chart(fig_corr, width="stretch")

        st.divider()

    # --- Error Details Table ---
    st.subheader("Misclassification Details")
    failures = eval_rep.get("failures", [])
    if failures:
        for f in failures:
            clip = Path(f.get("clip_path", "")).name
            gt_s = f.get("gt_severity", "?")
            pred_s = f.get("pred_severity", "?")
            conf = f.get("confidence", 0)
            reason = f.get("causal_reasoning", "")[:200]

            # Color-coded expander
            with st.expander(f"{clip}  |  GT: {gt_s} -> Pred: {pred_s}  |  Conf: {conf:.2f}"):
                st.markdown(f"*{reason}...*")
    else:
        st.success("No misclassifications!")
