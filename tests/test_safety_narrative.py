from __future__ import annotations

import json
from pathlib import Path

from autorisk.report.safety_narrative import generate_from_json, generate_safety_narrative


def test_generate_safety_narrative_empty() -> None:
    assert generate_safety_narrative([]) == "No analysis results available."


def test_generate_safety_narrative_contains_core_sections() -> None:
    cosmos_results = [
        {
            "candidate_rank": 1,
            "clip_path": "clips/candidate_001.mp4",
            "severity": "HIGH",
            "confidence": 0.8,
            "causal_reasoning": "Close cut-in by adjacent vehicle.",
            "hazards": [{"type": "cut_in", "actors": ["adjacent_vehicle"]}],
            "recommended_action": "Brake and increase following distance.",
            "parse_success": True,
        },
        {
            "candidate_rank": 2,
            "clip_path": "clips/candidate_002.mp4",
            "severity": "LOW",
            "confidence": 0.6,
            "causal_reasoning": "Normal traffic flow.",
            "hazards": [],
            "parse_success": True,
        },
    ]

    text = generate_safety_narrative(cosmos_results)
    assert "# Safety Narrative Report" in text
    assert "## Severity Distribution" in text
    assert "- **HIGH** (Critical): 1 (50.0%)" in text
    assert "## Critical Incidents (HIGH)" in text
    assert "Brake and increase following distance." in text


def test_generate_from_json_writes_file(tmp_path: Path) -> None:
    results_path = tmp_path / "cosmos_results.json"
    out_path = tmp_path / "safety_narrative.md"
    results_path.write_text(
        json.dumps(
            [
                {
                    "candidate_rank": 1,
                    "severity": "LOW",
                    "causal_reasoning": "steady driving",
                    "parse_success": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    written = generate_from_json(results_path, out_path)
    assert written == out_path
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "Safety Narrative Report" in content

