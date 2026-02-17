from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.fixture()
def sample_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    results = [
        {
            "candidate_rank": 1,
            "clip_path": str(run_dir / "clips" / "candidate_001.mp4"),
            "peak_time_sec": 10.0,
            "fused_score": 0.82,
            "severity": "HIGH",
            "hazards": [],
            "causal_reasoning": "test",
            "short_term_prediction": "",
            "recommended_action": "",
            "evidence": [],
            "confidence": 0.2,
            "parse_success": True,
            "error": "",
            "raw_answer": "{}",
        },
        {
            "candidate_rank": 2,
            "clip_path": str(run_dir / "clips" / "candidate_002.mp4"),
            "peak_time_sec": 20.0,
            "fused_score": 0.41,
            "severity": "LOW",
            "hazards": [],
            "causal_reasoning": "test",
            "short_term_prediction": "",
            "recommended_action": "",
            "evidence": [],
            "confidence": 0.9,
            "parse_success": False,
            "error": "parse failed",
            "raw_answer": "",
        },
    ]
    _write_json(run_dir / "cosmos_results.json", results)

    candidates_csv = (
        "rank,peak_time_sec,start_sec,end_sec,fused_score,clip_path,signal_scores\n"
        f"1,10.0,5.0,15.0,0.82,{run_dir / 'clips' / 'candidate_001.mp4'},\"{{'audio':0.2,'motion':0.8,'proximity':0.5}}\"\n"
        f"2,20.0,15.0,25.0,0.41,{run_dir / 'clips' / 'candidate_002.mp4'},\"{{'audio':0.3,'motion':0.1,'proximity':0.2}}\"\n"
    )
    (run_dir / "candidates.csv").write_text(candidates_csv, encoding="utf-8")

    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    (clips_dir / "candidate_001.mp4").write_bytes(b"fake-mp4-001")
    (clips_dir / "candidate_002.mp4").write_bytes(b"fake-mp4-002")

    return run_dir
