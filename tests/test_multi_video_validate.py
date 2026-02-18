from __future__ import annotations

import json
from pathlib import Path

from autorisk.multi_video.validate import (
    validate_multi_video_run_summary,
    validate_submission_metrics,
)


def test_validate_run_summary_happy_path(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "started_at_utc": "2026-02-18T00:00:00+00:00",
        "finished_at_utc": "2026-02-18T00:00:02+00:00",
        "elapsed_sec": 2.0,
        "dry_run": True,
        "resume": True,
        "fail_fast": False,
        "skip": {
            "supplement": True,
            "ttc": True,
            "grounding": True,
            "report": True,
        },
        "sources": [
            {
                "name": "public",
                "config_path": "configs/public.yaml",
                "output_dir": "outputs/public_run",
                "default_video_path": "data/public.mp4",
                "started_at_utc": "2026-02-18T00:00:00+00:00",
                "finished_at_utc": "2026-02-18T00:00:01+00:00",
                "elapsed_sec": 1.0,
                "ok": True,
                "clips_total": 2,
                "results_done_before": 2,
                "results_done_after_infer": 2,
                "clips_total_after_infer": 2,
                "steps": [
                    {
                        "label": "public/infer",
                        "ok": True,
                        "returncode": 0,
                        "elapsed_sec": 0.0,
                        "skipped": True,
                        "reason": "resume: already complete",
                    }
                ],
            }
        ],
        "ok": True,
        "failed_sources": 0,
    }
    path = tmp_path / "run_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    res = validate_multi_video_run_summary(path)
    assert res.ok is True
    assert res.issues == []


def test_validate_run_summary_semantic_mismatch(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "started_at_utc": "x",
        "finished_at_utc": "y",
        "elapsed_sec": 1.0,
        "dry_run": False,
        "resume": True,
        "fail_fast": False,
        "skip": {
            "supplement": False,
            "ttc": False,
            "grounding": False,
            "report": False,
        },
        "sources": [
            {
                "name": "public",
                "config_path": "configs/public.yaml",
                "output_dir": "outputs/public_run",
                "default_video_path": "",
                "started_at_utc": "x",
                "finished_at_utc": "y",
                "elapsed_sec": 1.0,
                "ok": False,
                "clips_total": 1,
                "results_done_before": 0,
                "results_done_after_infer": 0,
                "clips_total_after_infer": 1,
                "steps": [
                    {
                        "label": "public/infer",
                        "ok": True,
                        "returncode": 0,
                        "elapsed_sec": 1.0,
                    }
                ],
            }
        ],
        "ok": True,
        "failed_sources": 0,
    }
    path = tmp_path / "run_summary_bad.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    res = validate_multi_video_run_summary(path)
    assert res.ok is False
    assert any("failed_sources mismatch" in issue.detail for issue in res.issues)
    assert any("source marked failed but no failed step exists" in issue.detail for issue in res.issues)


def test_validate_submission_metrics_semantic_mismatch(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "generated_at_utc": "2026-02-18T00:00:00+00:00",
        "sources_total": 1,
        "sources_available": 1,
        "clips_total": 10,
        "sources": [
            {
                "name": "public",
                "config_path": "configs/public.yaml",
                "output_dir": "outputs/public_run",
                "available": True,
                "clip_count": 3,
                "parse_success_count": 5,
                "parse_success_rate": 0.2,
                "severity_counts": {"NONE": 1, "LOW": 1, "MEDIUM": 1, "HIGH": 1},
                "accuracy": 0.3,
                "macro_f1": 0.3,
                "checklist_mean_total": 5.0,
                "mean_confidence": 0.5,
                "grounding_mean_score": 0.9,
                "ttc": {
                    "n_positive_min_ttc": 1,
                    "mean_min_ttc": 0.5,
                    "min_min_ttc": 0.5,
                    "spearman_rho_vs_severity": None,
                    "spearman_p_value": None,
                    "spearman_n": 2,
                },
                "fused_signal": {},
            }
        ],
    }
    path = tmp_path / "submission_metrics_bad.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    res = validate_submission_metrics(path)
    assert res.ok is False
    assert any("parse_success_count" in issue.detail for issue in res.issues)
    assert any("parse_success_rate mismatch" in issue.detail for issue in res.issues)
    assert any("severity_counts sum mismatch" in issue.detail for issue in res.issues)
    assert any("spearman_n" in issue.detail for issue in res.issues)


def test_validate_run_summary_rejects_unknown_schema_version(tmp_path: Path) -> None:
    payload = {
        "schema_version": 999,
        "started_at_utc": "2026-02-18T00:00:00+00:00",
        "finished_at_utc": "2026-02-18T00:00:01+00:00",
        "elapsed_sec": 1.0,
        "dry_run": True,
        "resume": True,
        "fail_fast": False,
        "skip": {"supplement": True, "ttc": True, "grounding": True, "report": True},
        "sources": [],
        "ok": True,
        "failed_sources": 0,
    }
    path = tmp_path / "run_summary_v999.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    res = validate_multi_video_run_summary(path)
    assert res.ok is False
    assert any("unsupported schema_version=999" in issue.detail for issue in res.issues)


def test_validate_submission_metrics_rejects_unknown_schema_version(tmp_path: Path) -> None:
    payload = {
        "schema_version": 999,
        "generated_at_utc": "2026-02-18T00:00:00+00:00",
        "sources_total": 0,
        "sources_available": 0,
        "clips_total": 0,
        "sources": [],
    }
    path = tmp_path / "submission_metrics_v999.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    res = validate_submission_metrics(path)
    assert res.ok is False
    assert any("unsupported schema_version=999" in issue.detail for issue in res.issues)
