from __future__ import annotations

import json
from pathlib import Path

from autorisk.multi_video import submission_metrics as sm


def test_ttc_correlation_has_expected_direction() -> None:
    cosmos_results = [
        {"clip_path": "clips/candidate_001.mp4", "severity": "NONE"},
        {"clip_path": "clips/candidate_002.mp4", "severity": "LOW"},
        {"clip_path": "clips/candidate_003.mp4", "severity": "MEDIUM"},
        {"clip_path": "clips/candidate_004.mp4", "severity": "HIGH"},
    ]
    ttc_results = [
        {"clip_path": "clips/candidate_001.mp4", "min_ttc": 4.0},
        {"clip_path": "clips/candidate_002.mp4", "min_ttc": 3.0},
        {"clip_path": "clips/candidate_003.mp4", "min_ttc": 2.0},
        {"clip_path": "clips/candidate_004.mp4", "min_ttc": 1.0},
    ]
    out = sm.ttc_correlation(cosmos_results, ttc_results)
    if sm.spearmanr is None:
        assert out["rho"] is None
        assert out["p_value"] is None
        assert out["n"] == 0
    else:
        assert out["n"] == 4
        assert out["rho"] is not None
        assert out["rho"] < 0


def test_write_submission_metrics_roundtrip(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "sources_total": 1,
        "sources_available": 1,
        "clips_total": 10,
        "sources": [{"name": "demo"}],
    }
    out_path = sm.write_submission_metrics(payload, output_path=tmp_path / "submission_metrics.json")
    assert out_path.exists()
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded == payload

