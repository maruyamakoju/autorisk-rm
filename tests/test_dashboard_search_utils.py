from __future__ import annotations

from autorisk.dashboard.search_utils import highlight_terms, parse_query_terms, search_clips


def test_parse_query_terms_deduplicates_and_normalizes() -> None:
    terms = parse_query_terms("  Pedestrian  pedestrian  Collision  ")
    assert terms == ["pedestrian", "collision"]


def test_highlight_terms_escapes_html() -> None:
    text = "<script>alert('x')</script> pedestrian"
    rendered = highlight_terms(text, ["pedestrian"])
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "<mark>pedestrian</mark>" in rendered


def test_search_clips_ranks_by_relevance() -> None:
    cosmos_results = [
        {
            "candidate_rank": 2,
            "severity": "LOW",
            "causal_reasoning": "Vehicle gently changes lane.",
            "evidence": ["lane marker visible"],
            "hazards": [],
        },
        {
            "candidate_rank": 1,
            "severity": "HIGH",
            "causal_reasoning": "Pedestrian suddenly enters path.",
            "evidence": ["close distance"],
            "hazards": [{"type": "pedestrian_crossing", "actors": ["pedestrian", "ego_vehicle"]}],
        },
    ]

    hits = search_clips(cosmos_results, "pedestrian high")
    assert len(hits) == 1
    assert hits[0].clip["candidate_rank"] == 1
    assert hits[0].relevance >= 7.0
    assert "severity" in hits[0].matched_fields
    assert "hazards" in hits[0].matched_fields

