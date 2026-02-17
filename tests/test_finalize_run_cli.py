from __future__ import annotations

import json
from pathlib import Path
import zipfile
import re

from click.testing import CliRunner
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from autorisk.audit.verify import verify_audit_pack
from autorisk.cli import cli
from autorisk.review.log import append_review_decision


_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def _write_keypair(tmp_path: Path, *, prefix: str = "") -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_path = tmp_path / f"{prefix}private_key.pem"
    public_path = tmp_path / f"{prefix}public_key.pem"
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


def _prepare_review_log(run_dir: Path) -> None:
    log_path = run_dir / "review_log.jsonl"
    append_review_decision(
        run_dir=run_dir,
        candidate_rank=1,
        severity_after="MEDIUM",
        reason="reviewed high-severity clip",
        log_path=log_path,
    )
    append_review_decision(
        run_dir=run_dir,
        candidate_rank=2,
        severity_after="LOW",
        reason="reviewed parse-failure clip",
        log_path=log_path,
    )


def test_finalize_run_audit_grade_accepts_keyring_anchor(sample_run_dir: Path, tmp_path: Path) -> None:
    _prepare_review_log(sample_run_dir)
    private_key, active_public_key = _write_keypair(tmp_path, prefix="active_")
    _, old_public_key = _write_keypair(tmp_path, prefix="old_")

    keyring_dir = tmp_path / "trusted_keys"
    keyring_dir.mkdir(parents=True, exist_ok=True)
    (keyring_dir / "active.pem").write_bytes(active_public_key.read_bytes())
    (keyring_dir / "old.pem").write_bytes(old_public_key.read_bytes())

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "finalize-run",
            "-r",
            str(sample_run_dir),
            "--policy",
            "configs/policy.yaml",
            "--no-include-clips",
            "--audit-grade",
            "--sign-private-key",
            str(private_key),
            "--sign-public-key-dir",
            str(keyring_dir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "[audit-grade]" in result.output
    assert "verifier-bundle" in result.output
    assert "audit-handoff" in result.output
    assert "audit-validate: issues=0" in result.output

    pack_dirs = sorted(p for p in sample_run_dir.glob("audit_pack_*") if p.is_dir())
    assert pack_dirs
    pack_dir = pack_dirs[-1]
    pack_zips = sorted(sample_run_dir.glob("audit_pack_*.zip"))
    assert pack_zips
    pack_zip = pack_zips[-1]

    verify_res = verify_audit_pack(
        pack_zip,
        public_key_dir=keyring_dir,
        require_signature=True,
        require_public_key=True,
        require_attestation=True,
    )
    assert verify_res.ok is True
    assert verify_res.signature_verified is True
    assert verify_res.attestation_verified is True

    finalize_record_path = sample_run_dir / "finalize_record.json"
    assert finalize_record_path.exists()
    finalize_record = json.loads(finalize_record_path.read_text(encoding="utf-8"))
    assert finalize_record["audit_grade"] is True
    assert finalize_record["pack_fingerprint"] == verify_res.checksums_sha256
    assert finalize_record["signature_key_id"] == verify_res.signature_key_id
    assert finalize_record["validate_ok"] is True
    assert finalize_record["validate_issues_count"] == 0
    assert finalize_record["handoff_path"] != ""
    assert len(str(finalize_record["handoff_checksums_sha256"])) == 64
    assert len(str(finalize_record["handoff_pack_zip_sha256"])) == 64
    assert len(str(finalize_record["handoff_verifier_bundle_zip_sha256"])) == 64

    packed_record_path = pack_dir / "run_artifacts" / "finalize_record.json"
    assert packed_record_path.exists()
    verifier_bundle_dir = sample_run_dir / "verifier_bundle"
    assert (verifier_bundle_dir / "VERIFY.md").exists()
    assert (verifier_bundle_dir / "keys" / "trusted").exists()
    handoff_dir = sample_run_dir / "handoff_latest"
    assert (handoff_dir / "PACK.zip").exists()
    assert (handoff_dir / "verifier_bundle.zip").exists()
    assert (handoff_dir / "finalize_record.json").exists()
    assert (handoff_dir / "audit_validate_report.json").exists()
    assert (handoff_dir / "HANDOFF.md").exists()
    assert (handoff_dir / "handoff_checksums.sha256.txt").exists()
    handoff_finalize_record = json.loads((handoff_dir / "finalize_record.json").read_text(encoding="utf-8"))
    for key in [
        "validate_ok",
        "validate_issues_count",
        "handoff_path",
        "handoff_checksums_sha256",
        "handoff_pack_zip_sha256",
        "handoff_verifier_bundle_zip_sha256",
    ]:
        assert handoff_finalize_record[key] == finalize_record[key]

    with zipfile.ZipFile(pack_zip, "r") as zf:
        members = set(zf.namelist())
        assert "run_artifacts/audit_validate_report.json" in members
        assert "run_artifacts/finalize_record.json" in members
        packed_finalize = json.loads(zf.read("run_artifacts/finalize_record.json").decode("utf-8"))

    # PACK-internal finalize record keeps handoff anchors and avoids circular handoff hashes.
    assert _HEX64.fullmatch(str(packed_finalize.get("handoff_anchor_checksums_sha256", "")))
    assert _HEX64.fullmatch(str(packed_finalize.get("handoff_anchor_verifier_bundle_zip_sha256", "")))
    assert str(packed_finalize.get("handoff_pack_zip_sha256", "")) == ""
    assert str(packed_finalize.get("handoff_checksums_sha256", "")) == ""


def test_finalize_run_audit_grade_requires_trusted_anchor(sample_run_dir: Path, tmp_path: Path) -> None:
    _prepare_review_log(sample_run_dir)
    private_key, _ = _write_keypair(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "finalize-run",
            "-r",
            str(sample_run_dir),
            "--policy",
            "configs/policy.yaml",
            "--no-include-clips",
            "--audit-grade",
            "--sign-private-key",
            str(private_key),
        ],
    )

    assert result.exit_code != 0
    assert "--audit-grade requires --sign-public-key or --sign-public-key-dir" in result.output
    assert not any(sample_run_dir.glob("audit_pack_*"))


def test_finalize_run_audit_grade_requires_sign_private_key(sample_run_dir: Path, tmp_path: Path) -> None:
    _prepare_review_log(sample_run_dir)
    _, public_key = _write_keypair(tmp_path, prefix="anchor_")
    keyring_dir = tmp_path / "trusted_keys_only_public"
    keyring_dir.mkdir(parents=True, exist_ok=True)
    (keyring_dir / "active.pem").write_bytes(public_key.read_bytes())

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "finalize-run",
            "-r",
            str(sample_run_dir),
            "--policy",
            "configs/policy.yaml",
            "--no-include-clips",
            "--audit-grade",
            "--sign-public-key-dir",
            str(keyring_dir),
        ],
    )

    assert result.exit_code != 0
    assert "--audit-grade requires --sign-private-key" in result.output
    assert not any(sample_run_dir.glob("audit_pack_*"))
