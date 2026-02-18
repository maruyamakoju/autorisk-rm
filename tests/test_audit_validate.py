from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from omegaconf import OmegaConf

from autorisk.audit.pack import build_audit_pack
from autorisk.audit.sign import sign_audit_pack
from autorisk.audit.validate import validate_audit_pack
from autorisk.cli import cli
from autorisk.review.log import append_review_decision, apply_review_overrides


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


def _write_ed25519_keypair(tmp_path: Path) -> tuple[Path, Path]:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_path = tmp_path / "private_key.pem"
    public_path = tmp_path / "public_key.pem"
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path


def test_audit_validate_happy_path_for_dir_and_zip(sample_run_dir: Path) -> None:
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=True,
    )
    assert pack_res.zip_path is not None

    dir_res = validate_audit_pack(pack_res.output_dir)
    zip_res = validate_audit_pack(pack_res.zip_path)
    assert dir_res.ok is True
    assert zip_res.ok is True
    assert dir_res.issues == []
    assert zip_res.issues == []

    runner = CliRunner()
    cli_res = runner.invoke(
        cli,
        ["audit-validate", "-p", str(pack_res.zip_path), "--enforce"],
        catch_exceptions=False,
    )
    assert cli_res.exit_code == 0, cli_res.output
    assert "Issues: 0" in cli_res.output


def test_audit_validate_audit_grade_profile_requires_artifacts(sample_run_dir: Path) -> None:
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=False,
    )
    result = validate_audit_pack(
        pack_res.output_dir,
        profile="audit-grade",
        semantic_checks=False,
    )
    assert result.ok is False
    missing = {issue.path for issue in result.issues if issue.kind == "missing_file"}
    assert "signature.json" in missing
    assert "run_artifacts/finalize_record.json" in missing
    assert "run_artifacts/audit_validate_report.json" in missing
    assert "run_artifacts/run_summary.json" in missing
    assert "run_artifacts/submission_metrics.json" in missing
    assert "run_artifacts/policy_snapshot.json" in missing
    assert "run_artifacts/review_apply_report.json" in missing
    assert "run_artifacts/review_diff_report.json" in missing
    assert "run_artifacts/cosmos_results_reviewed.json" in missing

    runner = CliRunner()
    cli_res = runner.invoke(
        cli,
        [
            "audit-validate",
            "-p",
            str(pack_res.output_dir),
            "--profile",
            "audit-grade",
            "--enforce",
        ],
        catch_exceptions=False,
    )
    assert cli_res.exit_code == 2
    assert "signature.json" in cli_res.output


def test_audit_validate_detects_duplicate_candidate_rank(sample_run_dir: Path) -> None:
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=False,
    )
    trace_path = pack_res.output_dir / "decision_trace.jsonl"
    lines = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip() != ""]
    assert len(lines) >= 1
    trace_path.write_text("\n".join([lines[0], lines[0]]) + "\n", encoding="utf-8")

    result = validate_audit_pack(pack_res.output_dir, semantic_checks=True)
    assert result.ok is False
    assert any(
        issue.kind == "semantic_error" and "duplicate candidate_rank" in issue.detail
        for issue in result.issues
    )

    runner = CliRunner()
    cli_res = runner.invoke(
        cli,
        ["audit-validate", "-p", str(pack_res.output_dir), "--enforce"],
        catch_exceptions=False,
    )
    assert cli_res.exit_code == 2
    assert "duplicate candidate_rank" in cli_res.output


def test_audit_validate_detects_review_record_hash_mismatch(sample_run_dir: Path) -> None:
    review_log = sample_run_dir / "review_log.jsonl"
    append_review_decision(
        run_dir=sample_run_dir,
        candidate_rank=1,
        severity_after="MEDIUM",
        reason="manual review",
        log_path=review_log,
    )
    apply_review_overrides(
        run_dir=sample_run_dir,
        log_path=review_log,
        output_path=None,
        allow_stale=False,
        write_report=True,
    )
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        review_log=review_log,
        include_clips=False,
        create_zip=False,
    )

    reviewed_path = pack_res.output_dir / "run_artifacts" / "cosmos_results_reviewed.json"
    reviewed_obj = json.loads(reviewed_path.read_text(encoding="utf-8"))
    reviewed_obj[0]["review"]["record_sha256"] = "0" * 64
    reviewed_path.write_text(json.dumps(reviewed_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    res = validate_audit_pack(pack_res.output_dir, semantic_checks=True)
    assert res.ok is False
    assert any(
        issue.kind == "semantic_error" and "record_sha256 not found in review_log" in issue.detail
        for issue in res.issues
    )


def test_audit_validate_detects_finalize_signature_key_mismatch(sample_run_dir: Path, tmp_path: Path) -> None:
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=False,
    )
    private_key, public_key = _write_ed25519_keypair(tmp_path)
    sign_audit_pack(
        pack_res.output_dir,
        private_key_path=private_key,
        public_key_path=public_key,
    )

    finalize_record_path = pack_res.output_dir / "run_artifacts" / "finalize_record.json"
    finalize_record_path.parent.mkdir(parents=True, exist_ok=True)
    finalize_record_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at_utc": "2026-02-16T00:00:00+00:00",
                "run_dir": str(sample_run_dir),
                "pack_dir": str(pack_res.output_dir),
                "zip_path": "",
                "pack_fingerprint": pack_res.checksums_sha256,
                "signature_present": True,
                "signature_verified": True,
                "signature_key_id": "deadbeefdeadbeef",
                "signature_key_source": "test",
                "policy_source": {},
                "policy_sha256": "",
                "revocation_file": "",
                "revocation_file_sha256": "",
                "revoked_key_ids_count": 0,
                "verification_issues": 0,
                "enforce": True,
                "audit_grade": False,
                "require_signature": True,
                "require_trusted_key": False,
                "trust_embedded_public_key": False,
                "verifier_bundle_path": "",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    res = validate_audit_pack(pack_res.output_dir, semantic_checks=True)
    assert res.ok is False
    assert any(
        issue.kind == "semantic_error" and "signature_key_id mismatch" in issue.detail
        for issue in res.issues
    )


def test_audit_validate_rejects_pack_finalize_handoff_hashes(sample_run_dir: Path) -> None:
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=False,
    )

    finalize_record_path = pack_res.output_dir / "run_artifacts" / "finalize_record.json"
    finalize_record_path.parent.mkdir(parents=True, exist_ok=True)
    finalize_record_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at_utc": "2026-02-16T00:00:00+00:00",
                "run_dir": str(sample_run_dir),
                "pack_dir": str(pack_res.output_dir),
                "zip_path": "",
                "pack_fingerprint": pack_res.checksums_sha256,
                "signature_present": False,
                "signature_verified": None,
                "signature_key_id": "",
                "signature_key_source": "",
                "policy_source": {},
                "policy_sha256": "",
                "revocation_file": "",
                "revocation_file_sha256": "",
                "revoked_key_ids_count": 0,
                "verification_issues": 0,
                "enforce": True,
                "audit_grade": False,
                "require_signature": False,
                "require_trusted_key": False,
                "trust_embedded_public_key": False,
                "verifier_bundle_path": "",
                "handoff_path": "outputs/handoff_latest",
                "handoff_checksums_sha256": "a" * 64,
                "handoff_pack_zip_sha256": "b" * 64,
                "handoff_verifier_bundle_zip_sha256": "c" * 64,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    res = validate_audit_pack(pack_res.output_dir, semantic_checks=True)
    assert res.ok is False
    assert any(
        issue.kind == "semantic_error"
        and "PACK-internal run_artifacts/finalize_record.json must not include handoff_* hashes" in issue.detail
        for issue in res.issues
    )


def test_audit_pack_sanitizes_finalize_record_for_pack_contract(sample_run_dir: Path) -> None:
    run_finalize = sample_run_dir / "finalize_record.json"
    run_finalize.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at_utc": "2026-02-16T00:00:00+00:00",
                "run_dir": str(sample_run_dir),
                "pack_dir": str(sample_run_dir / "placeholder_pack"),
                "zip_path": "",
                "pack_fingerprint": "f" * 64,
                "signature_present": False,
                "signature_verified": None,
                "signature_key_id": "",
                "signature_key_source": "",
                "policy_source": {},
                "policy_sha256": "",
                "revocation_file": "",
                "revocation_file_sha256": "",
                "revoked_key_ids_count": 0,
                "verification_issues": 0,
                "enforce": True,
                "audit_grade": False,
                "require_signature": False,
                "require_trusted_key": False,
                "trust_embedded_public_key": False,
                "verifier_bundle_path": "",
                "validate_ok": True,
                "validate_issues_count": 0,
                "validate_report_path": "",
                "validate_report_sha256": "",
                "handoff_path": str(sample_run_dir / "handoff_latest"),
                "handoff_checksums_sha256": "a" * 64,
                "handoff_pack_zip_sha256": "b" * 64,
                "handoff_verifier_bundle_zip_sha256": "c" * 64,
                "handoff_anchor_checksums_sha256": "d" * 64,
                "handoff_anchor_verifier_bundle_zip_sha256": "e" * 64,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=False,
    )
    packed_finalize_path = pack_res.output_dir / "run_artifacts" / "finalize_record.json"
    assert packed_finalize_path.exists()
    packed_finalize = json.loads(packed_finalize_path.read_text(encoding="utf-8"))
    assert packed_finalize["handoff_path"] == ""
    assert packed_finalize["handoff_checksums_sha256"] == ""
    assert packed_finalize["handoff_pack_zip_sha256"] == ""
    assert packed_finalize["handoff_verifier_bundle_zip_sha256"] == ""
    assert packed_finalize["handoff_anchor_checksums_sha256"] == "d" * 64
    assert packed_finalize["handoff_anchor_verifier_bundle_zip_sha256"] == "e" * 64

    # align fingerprint for semantic check; sanitize behavior itself is the contract under test
    packed_finalize["pack_fingerprint"] = pack_res.checksums_sha256
    packed_finalize_path.write_text(json.dumps(packed_finalize, ensure_ascii=False, indent=2), encoding="utf-8")
    res = validate_audit_pack(pack_res.output_dir, semantic_checks=True)
    assert res.ok is True


def test_audit_validate_applies_multi_video_semantics_for_run_summary(sample_run_dir: Path) -> None:
    run_summary_path = sample_run_dir / "run_summary.json"
    run_summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "started_at_utc": "2026-02-18T00:00:00+00:00",
                "finished_at_utc": "2026-02-18T00:00:01+00:00",
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
                        "output_dir": str(sample_run_dir),
                        "default_video_path": "",
                        "started_at_utc": "2026-02-18T00:00:00+00:00",
                        "finished_at_utc": "2026-02-18T00:00:01+00:00",
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
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=False,
    )
    res = validate_audit_pack(pack_res.output_dir, semantic_checks=True)
    assert res.ok is False
    assert any(
        issue.kind == "semantic_error"
        and issue.path == "run_artifacts/run_summary.json"
        and "failed_sources mismatch" in issue.detail
        for issue in res.issues
    )
