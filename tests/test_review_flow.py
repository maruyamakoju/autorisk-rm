from __future__ import annotations

import hashlib
import json
from pathlib import Path

from autorisk.review.log import append_review_decision, apply_review_overrides


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_review_approve_records_results_sha(sample_run_dir: Path) -> None:
    log_path = sample_run_dir / "review_log.jsonl"
    path, record, rec_sha = append_review_decision(
        run_dir=sample_run_dir,
        candidate_rank=1,
        severity_after="MEDIUM",
        reason="manual review escalation",
        evidence_refs=["frame:120"],
        log_path=log_path,
    )

    assert path == log_path.resolve()
    assert rec_sha != ""
    assert record["decision_before"]["severity"] == "HIGH"
    assert record["decision_after"]["severity"] == "MEDIUM"
    assert record["run"]["results_sha256"] == _sha256_file(sample_run_dir / "cosmos_results.json")


def test_review_apply_updates_severity_and_attaches_metadata(sample_run_dir: Path) -> None:
    log_path = sample_run_dir / "review_log.jsonl"
    append_review_decision(
        run_dir=sample_run_dir,
        candidate_rank=1,
        severity_after="MEDIUM",
        reason="manual review escalation",
        evidence_refs=["frame:120"],
        log_path=log_path,
    )

    res = apply_review_overrides(
        run_dir=sample_run_dir,
        log_path=log_path,
        output_path=None,
        allow_stale=False,
        write_report=True,
    )

    assert res.applied == 1
    reviewed = json.loads((sample_run_dir / "cosmos_results_reviewed.json").read_text(encoding="utf-8"))
    first = reviewed[0]
    assert first["severity"] == "MEDIUM"
    assert "review" in first
    assert first["review"]["decision_after"]["severity"] == "MEDIUM"
    assert first["review"]["record_sha256"] != ""
    assert res.diff_report_path.exists()
    diff_report = json.loads(res.diff_report_path.read_text(encoding="utf-8"))
    assert diff_report["override_count"] == 1
    assert diff_report["transition_histogram"]["HIGH->MEDIUM"] == 1
