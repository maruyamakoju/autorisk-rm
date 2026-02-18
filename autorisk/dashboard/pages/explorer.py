"""Clip Explorer - per-clip deep dive with video, saliency, signals, VLM output."""

from __future__ import annotations

import base64
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from autorisk.dashboard.comparison_utils import SEVERITY_COLORS, SEVERITY_ORDER


def _severity_badge_html(severity: str) -> str:
    bg_map = {
        "NONE": "background:#374151; color:#D1D5DB;",
        "LOW": "background:#065F46; color:#6EE7B7;",
        "MEDIUM": "background:#78350F; color:#FCD34D;",
        "HIGH": "background:#7F1D1D; color:#FCA5A5;",
    }
    style = bg_map.get(severity, bg_map["NONE"])
    return (
        f'<span style="{style} padding:4px 14px; border-radius:16px; '
        f'font-weight:700; font-size:15px;">{severity}</span>'
    )


def render(data: dict) -> None:
    cosmos = data["cosmos_results"]
    candidates = data["candidates"]
    gt = data["gt_labels"]
    ttc_lookup = data["ttc_lookup"]
    grounding_lookup = data["grounding_lookup"]
    saliency_images = data["saliency_images"]
    saliency_results = data["saliency_results"]
    clips_dir = Path(data["clips_dir"])

    if not cosmos:
        st.warning("No Cosmos results found.")
        return

    # --- Clip Selector ---
    clip_options = []
    for r in cosmos:
        rank = r["candidate_rank"]
        sev = r.get("severity", "NONE")
        clip_name = Path(r["clip_path"]).name
        gt_sev = gt.get(clip_name, "") if gt else ""
        gt_str = f"  (GT: {gt_sev})" if gt_sev else ""
        label = f"#{rank:02d}  |  {sev}{gt_str}  |  {clip_name}"
        clip_options.append((label, rank))

    selected_label = st.selectbox(
        "Select Clip",
        [opt[0] for opt in clip_options],
        index=0,
    )
    selected_rank = clip_options[[opt[0] for opt in clip_options].index(selected_label)][1]

    # Find selected clip data
    result = next(r for r in cosmos if r["candidate_rank"] == selected_rank)
    clip_name = Path(result["clip_path"]).name
    candidate = next((c for c in candidates if c["rank"] == selected_rank), None)
    gt_sev = gt.get(clip_name, "N/A")

    st.divider()

    # --- Main Layout ---
    col_video, col_info = st.columns([1, 1])

    with col_video:
        # Video player
        clip_path = clips_dir / clip_name
        if clip_path.exists():
            st.video(str(clip_path))
        else:
            st.warning(f"Video file not found: {clip_path}")

        # Saliency heatmap
        sal_data = saliency_images.get(clip_name, {})
        if sal_data:
            st.markdown("**Gradient Saliency**")
            sal_col1, sal_col2 = st.columns(2)
            with sal_col1:
                if sal_data.get("raw_frame_b64"):
                    raw_bytes = base64.b64decode(sal_data["raw_frame_b64"])
                    st.image(raw_bytes, caption="Raw Frame", width="stretch")
            with sal_col2:
                if sal_data.get("heatmap_b64"):
                    heat_bytes = base64.b64decode(sal_data["heatmap_b64"])
                    st.image(heat_bytes, caption="Attention Heatmap", width="stretch")

        # Temporal saliency bar (attention over time)
        sal_result = next(
            (
                s
                for s in saliency_results
                if isinstance(s, dict) and str(s.get("clip_name", "")) == clip_name
            ),
            None,
        )
        if sal_result and sal_result.get("frame_saliency_sums"):
            st.markdown("**Temporal Attention**")
            frame_scores = []
            for raw in sal_result["frame_saliency_sums"]:
                try:
                    frame_scores.append(float(raw))
                except Exception:
                    frame_scores.append(0.0)
            n_frames = len(frame_scores)

            # Normalize scores to 0-1 range for coloring
            max_score = max(frame_scores) if frame_scores else 1
            normalized = [s / max_score for s in frame_scores]

            # Create color gradient: blue (low) -> yellow -> red (high)
            colors = []
            for norm_val in normalized:
                if norm_val < 0.5:
                    # blue -> yellow
                    r = int(norm_val * 2 * 255)
                    g = int(norm_val * 2 * 255)
                    b = int((1 - norm_val * 2) * 255)
                else:
                    # yellow -> red
                    r = 255
                    g = int((2 - norm_val * 2) * 255)
                    b = 0
                colors.append(f"rgb({r},{g},{b})")

            # Time labels (assuming ~10s clip divided into N frames)
            clip_duration = 10.0  # seconds (typical clip duration)
            frame_times = [f"{i * clip_duration / n_frames:.1f}s" for i in range(n_frames)]

            fig = go.Figure(data=[
                go.Bar(
                    x=frame_times,
                    y=frame_scores,
                    marker=dict(color=colors, line=dict(width=0)),
                    hovertemplate="Time: %{x}<br>Attention: %{y:.2f}<extra></extra>",
                )
            ])
            fig.update_layout(
                height=180,
                margin=dict(l=10, r=10, t=10, b=40),
                xaxis_title="Time in Clip",
                yaxis_title="Attention",
                showlegend=False,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_info:
        # Severity badges
        st.markdown(
            f"**Predicted:** {_severity_badge_html(result.get('severity', 'NONE'))} "
            f"&nbsp;&nbsp; **Ground Truth:** {_severity_badge_html(gt_sev)}",
            unsafe_allow_html=True,
        )
        conf = result.get("confidence", 0)
        score = result.get("fused_score", 0)
        st.markdown(f"**Fused Score:** `{score:.3f}` &nbsp;&nbsp; **Confidence:** `{conf:.2f}`")

        # Voting badge (if metadata exists)
        vote_dist = result.get("vote_distribution")
        if vote_dist and isinstance(vote_dist, dict):
            _render_vote_badge(vote_dist)

        st.markdown("---")

        # Hazards
        hazards = result.get("hazards", [])
        if hazards:
            st.markdown("**Hazards**")
            for h in hazards:
                actors = ", ".join(h.get("actors", []))
                st.markdown(
                    f'- **{h.get("type", "Unknown")}**: {actors}\n'
                    f'  - _{h.get("spatial_relation", "")}_'
                )
        else:
            st.markdown("**Hazards:** None identified")

        st.markdown("---")

        # Causal Reasoning
        causal = result.get("causal_reasoning", "")
        if causal:
            st.markdown("**Causal Reasoning**")
            st.info(causal)

        # Prediction
        pred = result.get("short_term_prediction", "")
        if pred:
            st.markdown("**Short-term Prediction**")
            st.warning(pred)

        # Recommended Action
        action = result.get("recommended_action", "")
        if action:
            st.markdown("**Recommended Action**")
            st.success(action)

    st.divider()

    # --- Bottom row: Signals + TTC ---
    col_sig, col_ttc = st.columns([1, 1])

    with col_sig:
        st.markdown("**Signal Scores**")
        if candidate:
            scores = candidate["signal_scores"]

            # Radar chart
            categories = list(scores.keys())
            values = list(scores.values())
            values_closed = values + [values[0]]
            categories_closed = categories + [categories[0]]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                fillcolor="rgba(124, 58, 237, 0.2)",
                line=dict(color="#7C3AED", width=2),
                name="Signal Score",
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

        # Grounding info
        g_data = grounding_lookup.get(clip_name, {})
        if g_data:
            gscore = g_data.get("grounding_score", 0)
            active = g_data.get("active_signals", [])
            mentioned = g_data.get("mentioned_signals", [])
            ungrounded = g_data.get("ungrounded_signals", [])

            st.markdown(f"**Cross-Modal Grounding:** `{gscore:.0%}`")
            st.caption(
                f"Active: {', '.join(active) or 'none'} | "
                f"VLM Mentioned: {', '.join(mentioned) or 'none'} | "
                f"Ungrounded: {', '.join(ungrounded) or 'none'}"
            )

    with col_ttc:
        st.markdown("**Time-to-Collision (TTC)**")
        ttc_data = ttc_lookup.get(clip_name, {})
        if ttc_data:
            min_ttc = ttc_data.get("min_ttc", -1)
            n_tracks = ttc_data.get("n_tracks", 0)
            n_critical = ttc_data.get("n_critical", 0)
            ttc_timeline = ttc_data.get("ttc_per_second", [])

            m1, m2, m3 = st.columns(3)
            m1.metric("Min TTC", f"{min_ttc:.2f}s" if min_ttc > 0 else "N/A")
            m2.metric("Tracks", n_tracks)
            m3.metric("Critical", n_critical)

            # TTC timeline chart
            if ttc_timeline:
                valid_ttc = [(i, v) for i, v in enumerate(ttc_timeline) if v > 0]
                if valid_ttc:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=[v[0] for v in valid_ttc],
                        y=[v[1] for v in valid_ttc],
                        mode="lines+markers",
                        line=dict(color="#F59E0B", width=2),
                        marker=dict(size=6),
                        name="TTC",
                    ))
                    # Critical threshold line
                    fig2.add_hline(y=1.5, line_dash="dash", line_color="#EF4444",
                                  annotation_text="Critical (1.5s)")
                    fig2.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=250,
                        margin=dict(t=20, b=40, l=40, r=20),
                        xaxis_title="Second in Clip",
                        yaxis_title="TTC (seconds)",
                    )
                    st.plotly_chart(fig2, width="stretch")
                else:
                    st.caption("No valid TTC measurements in this clip.")

            # Track details
            tracks = ttc_data.get("tracks", [])
            if tracks:
                with st.expander(f"Track Details ({len(tracks)} objects)"):
                    for t in tracks:
                        st.markdown(
                            f"- **{t['class_name']}** (ID {t['track_id']}): "
                            f"min TTC={t['min_ttc']:.2f}s, "
                            f"mean TTC={t['mean_ttc']:.2f}s"
                        )
        else:
            st.caption("No TTC data available for this clip.")


def _render_vote_badge(vote_dist: dict) -> None:
    """Render a vote distribution badge if voting metadata is present."""
    cleaned_votes: dict[str, float] = {}
    for sev in SEVERITY_ORDER:
        try:
            cleaned_votes[sev] = float(vote_dist.get(sev, 0) or 0)
        except Exception:
            cleaned_votes[sev] = 0.0

    total_votes = sum(cleaned_votes.values())
    if total_votes == 0:
        return

    # Build vote bars
    vote_html = '<div style="margin-top:8px;"><strong>Vote Distribution:</strong><div style="display:flex; gap:4px; margin-top:4px;">'
    for sev in SEVERITY_ORDER:
        count = cleaned_votes.get(sev, 0)
        if count > 0:
            pct = (count / total_votes) * 100
            color = SEVERITY_COLORS.get(sev, "#6B7280")
            vote_html += (
                f'<div style="background:{color}; color:white; padding:4px 8px; border-radius:4px; '
                f'font-size:11px; font-weight:700;">{sev}: {int(round(count))} ({pct:.0f}%)</div>'
            )
    vote_html += '</div></div>'

    st.markdown(vote_html, unsafe_allow_html=True)
