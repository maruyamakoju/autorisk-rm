"""Search page for natural language lookup over clip outputs."""

from __future__ import annotations

import streamlit as st

from autorisk.dashboard.data_loader import SEVERITY_COLORS
from autorisk.dashboard.search_utils import highlight_terms, parse_query_terms, search_clips


def render(data: dict) -> None:
    """Render the dashboard search page."""
    cosmos = data["cosmos_results"]

    st.title("Natural Language Search")
    st.markdown("Search across all analyzed clips using keywords.")

    query = st.text_input(
        "Search query",
        placeholder='e.g., "pedestrian", "collision", "emergency braking"',
    )

    if not query.strip():
        st.info("Try: pedestrian, collision, lane change, braking")
        return

    results = search_clips(cosmos, query)
    if not results:
        st.warning(f"No clips found matching '{query}'")
        return

    st.success(f"Found **{len(results)}** clips")
    terms = parse_query_terms(query)

    for i, hit in enumerate(results, 1):
        clip = hit.clip
        clip_name = clip.get("clip_path", "").split("/")[-1] or f"Clip {i}"
        severity = clip.get("severity", "NONE")
        conf = clip.get("confidence", 0)

        with st.expander(
            f"**#{i}** {clip_name} | {severity} ({conf:.0%}) | "
            f"Matched: {', '.join(hit.matched_fields)}"
        ):
            color = SEVERITY_COLORS.get(severity, "#6B7280")
            st.markdown(
                f"<span style='background-color:{color}; color:white; "
                f"padding:4px 12px; border-radius:4px; font-weight:bold;'>"
                f"{severity}</span>",
                unsafe_allow_html=True,
            )

            st.markdown("**Reasoning:**")
            causal = clip.get("causal_reasoning", "N/A")
            st.markdown(highlight_terms(causal, terms), unsafe_allow_html=True)

            hazards = clip.get("hazards", [])
            if hazards:
                st.markdown("**Hazards:**")
                for hazard in hazards:
                    if not isinstance(hazard, dict):
                        continue
                    h_type = highlight_terms(hazard.get("type", ""), terms)
                    actors = highlight_terms(", ".join(hazard.get("actors", [])), terms)
                    st.markdown(f"- {h_type} | {actors}", unsafe_allow_html=True)

            evidence = clip.get("evidence", [])
            if evidence:
                st.markdown("**Evidence:**")
                for ev in evidence:
                    st.markdown(f"- {highlight_terms(ev, terms)}", unsafe_allow_html=True)

