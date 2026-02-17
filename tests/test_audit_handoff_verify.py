from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from click.testing import CliRunner
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from autorisk.audit.handoff_verify import verify_audit_handoff
from autorisk.cli import cli
from autorisk.review.log import append_review_decision


def _write_keypair(tmp_path: Path) -> tuple[Path, Path]:
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


def _prepare_review_log(run_dir: Path) -> None:
    review_log = run_dir / "review_log.jsonl"
    append_review_decision(
        run_dir=run_dir,
        candidate_rank=1,
        severity_after="MEDIUM",
        reason="reviewed high-severity clip",
        log_path=review_log,
    )
    append_review_decision(
        run_dir=run_dir,
        candidate_rank=2,
        severity_after="LOW",
        reason="reviewed parse-failure clip",
        log_path=review_log,
    )


def _build_audit_grade_handoff(sample_run_dir: Path, tmp_path: Path) -> Path:
    _prepare_review_log(sample_run_dir)
    private_key, public_key = _write_keypair(tmp_path)
    keyring_dir = tmp_path / "trusted_keys"
    keyring_dir.mkdir(parents=True, exist_ok=True)
    (keyring_dir / "active.pem").write_bytes(public_key.read_bytes())

    handoff_dir = sample_run_dir / "handoff_latest"
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
            "--zip",
            "--audit-grade",
            "--sign-private-key",
            str(private_key),
            "--sign-public-key-dir",
            str(keyring_dir),
            "--handoff-out",
            str(handoff_dir),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    return handoff_dir


def _rewrite_zip_member(
    zip_path: Path,
    *,
    member_suffix: str,
    mutate_bytes,
    delete: bool = False,
) -> None:
    with tempfile.TemporaryDirectory(prefix="autorisk-test-zip-rewrite-") as tmp_dir:
        tmp_zip = Path(tmp_dir) / zip_path.name
        found = False
        with zipfile.ZipFile(zip_path, "r") as src, zipfile.ZipFile(
            tmp_zip, "w", compression=zipfile.ZIP_DEFLATED
        ) as dst:
            for info in src.infolist():
                if info.is_dir():
                    dst.writestr(info, b"")
                    continue
                payload = src.read(info.filename)
                if info.filename.endswith(member_suffix):
                    found = True
                    if delete:
                        continue
                    payload = mutate_bytes(payload)
                dst.writestr(info, payload)
        assert found, f"member not found in zip: {member_suffix}"
        tmp_zip.replace(zip_path)


def test_verify_audit_handoff_happy_path(sample_run_dir: Path, tmp_path: Path) -> None:
    handoff_dir = _build_audit_grade_handoff(sample_run_dir, tmp_path)
    result = verify_audit_handoff(handoff_dir)
    assert result.ok is True
    assert result.audit_verify_ok is True
    assert result.audit_validate_ok is True
    assert result.bundled_validate_report_match is True
    assert result.issues == []


def test_audit_handoff_verify_cli_detects_tampered_handoff(sample_run_dir: Path, tmp_path: Path) -> None:
    handoff_dir = _build_audit_grade_handoff(sample_run_dir, tmp_path)

    with (handoff_dir / "verifier_bundle.zip").open("ab") as f:
        f.write(b"tampered")

    runner = CliRunner()
    out_json = tmp_path / "handoff_verify.json"
    result = runner.invoke(
        cli,
        [
            "audit-handoff-verify",
            "-d",
            str(handoff_dir),
            "--json-out",
            str(out_json),
            "--enforce",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 2
    assert "Issues:" in result.output
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert any(
        (issue.get("kind") == "mismatch" and issue.get("path") == "verifier_bundle.zip")
        or (
            issue.get("kind") == "finalize_record_error"
            and "handoff_anchor_verifier_bundle_zip_sha256 mismatch" in issue.get("detail", "")
        )
        for issue in payload.get("issues", [])
    )


def test_verify_audit_handoff_detects_finalize_record_hash_mismatch(sample_run_dir: Path, tmp_path: Path) -> None:
    handoff_dir = _build_audit_grade_handoff(sample_run_dir, tmp_path)
    finalize_path = handoff_dir / "finalize_record.json"
    payload = json.loads(finalize_path.read_text(encoding="utf-8"))
    payload["validate_report_sha256"] = "0" * 64
    finalize_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    result = verify_audit_handoff(handoff_dir)
    assert result.ok is False
    assert any(
        issue.kind == "finalize_record_error" and "handoff finalize copy diverges" in issue.detail
        for issue in result.issues
    )


def test_verify_audit_handoff_detects_validate_report_tamper(sample_run_dir: Path, tmp_path: Path) -> None:
    handoff_dir = _build_audit_grade_handoff(sample_run_dir, tmp_path)
    validate_path = handoff_dir / "audit_validate_report.json"
    report = json.loads(validate_path.read_text(encoding="utf-8"))
    report["ok"] = False
    validate_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    result = verify_audit_handoff(handoff_dir)
    assert result.ok is False
    assert any(
        issue.kind in {"finalize_record_error", "report_mismatch"}
        and (
            issue.path in {"audit_validate_report.json", "PACK.zip!run_artifacts/finalize_record.json"}
            or issue.path.startswith("PACK.zip!")
        )
        for issue in result.issues
    )


def test_verify_audit_handoff_rejects_zip_slip_entries(sample_run_dir: Path, tmp_path: Path) -> None:
    handoff_dir = _build_audit_grade_handoff(sample_run_dir, tmp_path)
    bundle_zip = handoff_dir / "verifier_bundle.zip"
    rewritten = tmp_path / "bundle_with_slip.zip"

    with zipfile.ZipFile(bundle_zip, "r") as src, zipfile.ZipFile(
        rewritten, "w", compression=zipfile.ZIP_DEFLATED
    ) as dst:
        for info in src.infolist():
            if info.is_dir():
                continue
            dst.writestr(info.filename, src.read(info.filename))
        dst.writestr("../evil.txt", b"owned")

    rewritten.replace(bundle_zip)

    result = verify_audit_handoff(handoff_dir)
    assert result.ok is False
    assert any(issue.kind == "security_error" for issue in result.issues)


def test_verify_audit_handoff_detects_pack_finalize_record_tamper(sample_run_dir: Path, tmp_path: Path) -> None:
    handoff_dir = _build_audit_grade_handoff(sample_run_dir, tmp_path)
    pack_zip = handoff_dir / "PACK.zip"

    def _mutate_finalize_record(payload: bytes) -> bytes:
        obj = json.loads(payload.decode("utf-8"))
        obj["audit_grade"] = False
        return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")

    _rewrite_zip_member(
        pack_zip,
        member_suffix="run_artifacts/finalize_record.json",
        mutate_bytes=_mutate_finalize_record,
    )

    result = verify_audit_handoff(handoff_dir)
    assert result.ok is False
    assert any(
        issue.kind == "audit_verify_error"
        and "attestation_error" in issue.detail
        and "finalize_record_sha256" in issue.detail
        for issue in result.issues
    )


def test_verify_audit_handoff_detects_attestation_signature_tamper(sample_run_dir: Path, tmp_path: Path) -> None:
    handoff_dir = _build_audit_grade_handoff(sample_run_dir, tmp_path)
    pack_zip = handoff_dir / "PACK.zip"

    def _mutate_attestation(payload: bytes) -> bytes:
        obj = json.loads(payload.decode("utf-8"))
        obj["signature"] = "AA=="
        return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")

    _rewrite_zip_member(
        pack_zip,
        member_suffix="attestation.json",
        mutate_bytes=_mutate_attestation,
    )

    result = verify_audit_handoff(handoff_dir)
    assert result.ok is False
    assert any(
        issue.kind == "audit_verify_error"
        and "attestation_error" in issue.detail
        and ("invalid signature" in issue.detail or "invalid base64 signature" in issue.detail)
        for issue in result.issues
    )


def test_verify_audit_handoff_requires_attestation(sample_run_dir: Path, tmp_path: Path) -> None:
    handoff_dir = _build_audit_grade_handoff(sample_run_dir, tmp_path)
    pack_zip = handoff_dir / "PACK.zip"

    _rewrite_zip_member(
        pack_zip,
        member_suffix="attestation.json",
        mutate_bytes=lambda b: b,
        delete=True,
    )

    result = verify_audit_handoff(handoff_dir, require_attestation=True)
    assert result.ok is False
    assert any(
        issue.kind == "audit_verify_error"
        and "attestation_error: attestation.json not found" in issue.detail
        for issue in result.issues
    )


def test_verify_audit_handoff_rejects_self_consistent_handoff_finalize_tamper(sample_run_dir: Path, tmp_path: Path) -> None:
    handoff_dir = _build_audit_grade_handoff(sample_run_dir, tmp_path)
    checksums_path = handoff_dir / "handoff_checksums.sha256.txt"
    checksums_path.write_text(
        checksums_path.read_text(encoding="utf-8") + "# attacker touched checksums\n",
        encoding="utf-8",
    )
    new_checksums_sha = hashlib.sha256(checksums_path.read_bytes()).hexdigest()
    handoff_finalize_path = handoff_dir / "finalize_record.json"
    handoff_finalize = json.loads(handoff_finalize_path.read_text(encoding="utf-8"))
    handoff_finalize["handoff_checksums_sha256"] = new_checksums_sha
    handoff_finalize_path.write_text(json.dumps(handoff_finalize, ensure_ascii=False, indent=2), encoding="utf-8")

    result = verify_audit_handoff(handoff_dir)
    assert result.ok is False
    assert any(
        issue.kind == "finalize_record_error"
        and "handoff_anchor_checksums_sha256 mismatch" in issue.detail
        for issue in result.issues
    )


def test_verify_audit_handoff_rejects_pack_zip_row_in_handoff_checksums(sample_run_dir: Path, tmp_path: Path) -> None:
    handoff_dir = _build_audit_grade_handoff(sample_run_dir, tmp_path)
    checksums_path = handoff_dir / "handoff_checksums.sha256.txt"
    pack_sha = hashlib.sha256((handoff_dir / "PACK.zip").read_bytes()).hexdigest()
    with checksums_path.open("a", encoding="utf-8") as f:
        f.write(f"{pack_sha}  PACK.zip\n")

    result = verify_audit_handoff(handoff_dir)
    assert result.ok is False
    assert any(
        issue.kind == "security_error"
        and issue.path == "PACK.zip"
        and "must not include" in issue.detail
        for issue in result.issues
    )
