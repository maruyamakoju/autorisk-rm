from __future__ import annotations

import json
from pathlib import Path

from autorisk.dashboard.data_loader import load_data


def _write_minimal_cosmos_results(run_dir: Path) -> None:
    (run_dir / "cosmos_results.json").write_text(
        json.dumps(
            [
                {
                    "candidate_rank": 1,
                    "clip_path": "clips/candidate_001.mp4",
                    "severity": "LOW",
                    "parse_success": True,
                    "confidence": 0.5,
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_dashboard_loader_handles_malformed_inputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "demo_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_minimal_cosmos_results(run_dir)
    (run_dir / "eval_report.json").write_text("{bad-json", encoding="utf-8")
    (run_dir / "ttc_results.json").write_text(json.dumps([{"bad": "entry"}]), encoding="utf-8")
    (run_dir / "grounding_report.json").write_text(
        json.dumps({"details": [{"clip_name": "candidate_001.mp4", "ok": True}, "bad-row"]}),
        encoding="utf-8",
    )
    (run_dir / "candidates.csv").write_text(
        "\n".join(
            [
                "rank,peak_time_sec,start_sec,end_sec,fused_score,clip_path,signal_scores",
                "1,10.0,5.0,15.0,0.82,clips/candidate_001.mp4,\"{'audio':0.1,'motion':0.2,'proximity':0.3}\"",
                "2,20.0,15.0,25.0,bad-score,clips/candidate_002.mp4,\"{'audio':0.1}\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_data(str(run_dir))
    assert isinstance(loaded, dict)
    assert len(loaded["cosmos_results"]) == 1
    assert len(loaded["candidates"]) == 1  # malformed row is skipped
    assert loaded["candidates"][0]["signal_scores"]["audio"] == 0.1
    assert loaded["ttc_lookup"] == {}  # malformed ttc row ignored
    assert loaded["grounding_lookup"]["candidate_001.mp4"]["ok"] is True
    assert loaded["load_warnings_count"] >= 2
    assert any("eval_report.json" in msg for msg in loaded["load_warnings"])
    assert any("invalid float field 'fused_score'" in msg for msg in loaded["load_warnings"])


def test_dashboard_loader_defaults_on_missing_optional_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "minimal_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_minimal_cosmos_results(run_dir)

    loaded = load_data(str(run_dir))
    assert loaded["eval_report"] == {}
    assert loaded["analysis_report"] == {}
    assert loaded["ablation_results"] == []
    assert loaded["candidates"] == []
    assert loaded["gt_labels"] == {}
    assert loaded["load_warnings_count"] == 0
