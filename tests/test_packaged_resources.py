from __future__ import annotations

import os
from pathlib import Path

from click.testing import CliRunner
from omegaconf import OmegaConf

from autorisk.audit.pack import build_audit_pack
from autorisk.cli import cli
from autorisk.review.log import append_review_decision


def _sample_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "cosmos": {
                "backend": "local",
                "local": {
                    "model_name": "nvidia/Cosmos-Reason2-8B",
                    "max_new_tokens": 64,
                    "temperature": 0.2,
                    "torch_dtype": "float16",
                },
                "local_fps": 4,
            }
        }
    )


def test_policy_and_validate_work_outside_repo_cwd(sample_run_dir: Path, tmp_path: Path) -> None:
    review_log = sample_run_dir / "review_log.jsonl"
    append_review_decision(
        run_dir=sample_run_dir,
        candidate_rank=1,
        severity_after="MEDIUM",
        reason="reviewed high-severity clip",
        log_path=review_log,
    )
    append_review_decision(
        run_dir=sample_run_dir,
        candidate_rank=2,
        severity_after="LOW",
        reason="reviewed parse-failure clip",
        log_path=review_log,
    )
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        review_log=review_log,
        include_clips=False,
        create_zip=False,
    )

    outside = tmp_path / "outside_repo"
    outside.mkdir(parents=True, exist_ok=True)

    prev_cwd = Path.cwd()
    runner = CliRunner()
    try:
        os.chdir(outside)
        policy_res = runner.invoke(
            cli,
            [
                "policy-check",
                "-r",
                str(sample_run_dir),
                "--review-log",
                str(review_log),
                "--enforce",
            ],
            catch_exceptions=False,
        )
        assert policy_res.exit_code == 0, policy_res.output

        validate_res = runner.invoke(
            cli,
            [
                "audit-validate",
                "-p",
                str(pack_res.output_dir),
                "--enforce",
            ],
            catch_exceptions=False,
        )
        assert validate_res.exit_code == 0, validate_res.output
        assert "Issues: 0" in validate_res.output
    finally:
        os.chdir(prev_cwd)

