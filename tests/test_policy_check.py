from __future__ import annotations

from pathlib import Path

from autorisk.policy.check import run_policy_check
from autorisk.review.log import append_review_decision


def test_policy_check_detects_missing_required_reviews(sample_run_dir: Path) -> None:
    res = run_policy_check(
        run_dir=sample_run_dir,
        review_log=sample_run_dir / "review_log.jsonl",
        allow_stale=False,
        write_outputs=True,
    )

    assert res.passed is False
    assert res.required_review_count == 2
    assert res.missing_review_count == 2
    assert res.report_path.exists()
    assert res.queue_path.exists()
    assert res.snapshot_path.exists()


def test_policy_check_passes_when_required_reviews_exist(sample_run_dir: Path) -> None:
    log_path = sample_run_dir / "review_log.jsonl"
    append_review_decision(
        run_dir=sample_run_dir,
        candidate_rank=1,
        severity_after="MEDIUM",
        reason="reviewed high-severity clip",
        log_path=log_path,
    )
    append_review_decision(
        run_dir=sample_run_dir,
        candidate_rank=2,
        severity_after="LOW",
        reason="reviewed parse-failure clip",
        log_path=log_path,
    )

    res = run_policy_check(
        run_dir=sample_run_dir,
        review_log=log_path,
        allow_stale=False,
        write_outputs=True,
    )

    assert res.passed is True
    assert res.missing_review_count == 0
    assert len(res.violations) == 0


def test_policy_check_respects_yaml_policy(sample_run_dir: Path, tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "\n".join(
            [
                "required_review_severities:",
                "  - HIGH",
                "require_parse_failure_review: false",
                "require_error_review: false",
                "allow_stale: false",
                "",
            ]
        ),
        encoding="utf-8",
    )

    res = run_policy_check(
        run_dir=sample_run_dir,
        policy_path=policy_path,
        review_log=sample_run_dir / "review_log.jsonl",
        write_outputs=True,
    )

    assert res.required_review_count == 1
    assert res.missing_review_count == 1
    assert res.policy_source["source_type"] == "file"
    assert res.policy_source["policy_path"] == str(policy_path.resolve())
