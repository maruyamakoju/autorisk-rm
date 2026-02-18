from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

from click.testing import CliRunner
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from omegaconf import OmegaConf

from autorisk.audit.attestation import attest_audit_pack
from autorisk.audit.pack import build_audit_pack
from autorisk.audit.sign import sign_audit_pack
from autorisk.audit.verify import verify_audit_pack
from autorisk.cli import cli


def test_audit_verify_happy_path_for_dir_and_zip(sample_run_dir: Path) -> None:
    cfg = OmegaConf.create(
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

    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=cfg,
        include_clips=False,
        create_zip=True,
    )
    assert pack_res.output_dir.exists()
    assert pack_res.zip_path is not None and pack_res.zip_path.exists()
    assert pack_res.checksums_sha256 != ""

    dir_verify = verify_audit_pack(pack_res.output_dir)
    zip_verify = verify_audit_pack(pack_res.zip_path)

    assert dir_verify.ok is True
    assert zip_verify.ok is True
    assert dir_verify.issues == []
    assert zip_verify.issues == []
    assert dir_verify.unchecked_files == []
    assert zip_verify.unchecked_files == []


def test_audit_verify_reports_optional_unchecked_files(sample_run_dir: Path) -> None:
    cfg = OmegaConf.create(
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
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=cfg,
        include_clips=False,
        create_zip=True,
    )
    assert pack_res.zip_path is not None

    extra = pack_res.output_dir / "run_artifacts" / "finalize_record.json"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text('{"schema_version":1}', encoding="utf-8")

    dir_verify = verify_audit_pack(pack_res.output_dir)
    assert dir_verify.ok is True
    assert "run_artifacts/finalize_record.json" in (dir_verify.unchecked_files or [])


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


def _write_finalize_artifacts(run_dir: Path) -> None:
    (run_dir / "finalize_record.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pack_fingerprint": "",
                "signature_present": True,
                "signature_key_id": "",
                "validate_ok": True,
                "validate_issues_count": 0,
                "validate_report_path": "run_artifacts/audit_validate_report.json",
                "validate_report_sha256": "",
                "handoff_path": "",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "audit_validate_report.json").write_text(
        json.dumps(
            {
                "source": str(run_dir),
                "mode": "dir",
                "pack_root": str(run_dir),
                "schema_dir": "package://autorisk.resources.schemas",
                "files_validated": 0,
                "records_validated": 0,
                "ok": True,
                "issues": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


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


def test_audit_verify_require_attestation_happy_path(sample_run_dir: Path, tmp_path: Path) -> None:
    _write_finalize_artifacts(sample_run_dir)
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=True,
    )
    assert pack_res.zip_path is not None

    private_key, public_key = _write_keypair(tmp_path)
    sign_audit_pack(pack_res.zip_path, private_key_path=private_key, public_key_path=public_key)
    attest_audit_pack(pack_res.zip_path, private_key_path=private_key, public_key_path=public_key)

    result = verify_audit_pack(
        pack_res.zip_path,
        public_key=public_key,
        require_signature=True,
        require_public_key=True,
        require_attestation=True,
    )
    assert result.ok is True
    assert result.attestation_present is True
    assert result.attestation_verified is True


def test_audit_verify_require_attestation_fails_when_missing(sample_run_dir: Path, tmp_path: Path) -> None:
    _write_finalize_artifacts(sample_run_dir)
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=True,
    )
    assert pack_res.zip_path is not None

    private_key, public_key = _write_keypair(tmp_path)
    sign_audit_pack(pack_res.zip_path, private_key_path=private_key, public_key_path=public_key)

    result = verify_audit_pack(
        pack_res.zip_path,
        public_key=public_key,
        require_signature=True,
        require_public_key=True,
        require_attestation=True,
    )
    assert result.ok is False
    assert any(issue.kind == "attestation_error" and "not found" in issue.detail for issue in result.issues)


def test_audit_verify_detects_attestation_mismatch_on_finalize_tamper(sample_run_dir: Path, tmp_path: Path) -> None:
    _write_finalize_artifacts(sample_run_dir)
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=True,
    )
    assert pack_res.zip_path is not None

    private_key, public_key = _write_keypair(tmp_path)
    sign_audit_pack(pack_res.zip_path, private_key_path=private_key, public_key_path=public_key)
    attest_audit_pack(pack_res.zip_path, private_key_path=private_key, public_key_path=public_key)

    def _mutate_finalize_record(payload: bytes) -> bytes:
        obj = json.loads(payload.decode("utf-8"))
        obj["validate_ok"] = False
        return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")

    _rewrite_zip_member(
        pack_res.zip_path,
        member_suffix="run_artifacts/finalize_record.json",
        mutate_bytes=_mutate_finalize_record,
    )

    result = verify_audit_pack(
        pack_res.zip_path,
        public_key=public_key,
        require_signature=True,
        require_public_key=True,
        require_attestation=True,
    )
    assert result.ok is False
    assert any(
        issue.kind == "attestation_error" and "finalize_record_sha256" in issue.detail
        for issue in result.issues
    )


def test_audit_verify_rejects_unknown_attestation_schema_version(sample_run_dir: Path, tmp_path: Path) -> None:
    _write_finalize_artifacts(sample_run_dir)
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=True,
    )
    assert pack_res.zip_path is not None

    private_key, public_key = _write_keypair(tmp_path)
    sign_audit_pack(pack_res.zip_path, private_key_path=private_key, public_key_path=public_key)
    attest_audit_pack(pack_res.zip_path, private_key_path=private_key, public_key_path=public_key)

    def _mutate_attestation(payload: bytes) -> bytes:
        obj = json.loads(payload.decode("utf-8"))
        obj["schema_version"] = 999
        return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")

    _rewrite_zip_member(
        pack_res.zip_path,
        member_suffix="attestation.json",
        mutate_bytes=_mutate_attestation,
    )

    result = verify_audit_pack(
        pack_res.zip_path,
        public_key=public_key,
        require_signature=True,
        require_public_key=True,
        require_attestation=True,
    )
    assert result.ok is False
    assert any(
        issue.kind == "attestation_error"
        and "unsupported attestation schema_version" in issue.detail
        for issue in result.issues
    )


def test_audit_verify_cli_profile_audit_grade_enforces_flags(sample_run_dir: Path, tmp_path: Path) -> None:
    _write_finalize_artifacts(sample_run_dir)
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=True,
    )
    assert pack_res.zip_path is not None

    private_key, public_key = _write_keypair(tmp_path)
    sign_audit_pack(
        pack_res.zip_path,
        private_key_path=private_key,
        public_key_path=public_key,
        include_public_key=True,
    )
    attest_audit_pack(
        pack_res.zip_path,
        private_key_path=private_key,
        public_key_path=public_key,
        include_public_key=True,
    )

    runner = CliRunner()
    res_missing_anchor = runner.invoke(
        cli,
        [
            "audit-verify",
            "-p",
            str(pack_res.zip_path),
            "--profile",
            "audit-grade",
        ],
    )
    assert res_missing_anchor.exit_code == 2
    assert "no --public-key/--public-key-dir was provided" in res_missing_anchor.output

    res_ok = runner.invoke(
        cli,
        [
            "audit-verify",
            "-p",
            str(pack_res.zip_path),
            "--profile",
            "audit-grade",
            "--public-key",
            str(public_key),
        ],
        catch_exceptions=False,
    )
    assert res_ok.exit_code == 0, res_ok.output
    assert "[audit-grade]" in res_ok.output
    assert "Attestation verified: True" in res_ok.output


def test_audit_verify_cli_profile_default_prints_diagnostics_warning(sample_run_dir: Path, tmp_path: Path) -> None:
    _write_finalize_artifacts(sample_run_dir)
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=True,
    )
    assert pack_res.zip_path is not None

    private_key, public_key = _write_keypair(tmp_path)
    sign_audit_pack(
        pack_res.zip_path,
        private_key_path=private_key,
        public_key_path=public_key,
        include_public_key=True,
    )
    attest_audit_pack(
        pack_res.zip_path,
        private_key_path=private_key,
        public_key_path=public_key,
        include_public_key=True,
    )

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "audit-verify",
            "-p",
            str(pack_res.zip_path),
            "--profile",
            "default",
            "--strict",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "crypto requirements are NOT enforced" in res.output
    assert "Do not use for audit-grade acceptance." in res.output
    assert "Use: audit-verify --profile audit-grade" in res.output


def test_audit_verify_profile_audit_grade_rejects_signature_attestation_key_mismatch(
    sample_run_dir: Path, tmp_path: Path
) -> None:
    _write_finalize_artifacts(sample_run_dir)
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=True,
    )
    assert pack_res.zip_path is not None

    key_a_dir = tmp_path / "key_a"
    key_b_dir = tmp_path / "key_b"
    key_a_dir.mkdir(parents=True, exist_ok=True)
    key_b_dir.mkdir(parents=True, exist_ok=True)
    private_a, public_a = _write_keypair(key_a_dir)
    private_b, public_b = _write_keypair(key_b_dir)

    sign_audit_pack(pack_res.zip_path, private_key_path=private_a, public_key_path=public_a)
    attest_audit_pack(pack_res.zip_path, private_key_path=private_b, public_key_path=public_b)

    keyring_dir = tmp_path / "trusted_keys"
    keyring_dir.mkdir(parents=True, exist_ok=True)
    (keyring_dir / "signer.pem").write_bytes(public_a.read_bytes())
    (keyring_dir / "attester.pem").write_bytes(public_b.read_bytes())

    default_verify = verify_audit_pack(
        pack_res.zip_path,
        public_key_dir=keyring_dir,
        require_signature=True,
        require_public_key=True,
        require_attestation=True,
    )
    assert default_verify.ok is True
    assert default_verify.signature_verified is True
    assert default_verify.attestation_verified is True

    runner = CliRunner()
    out_json = tmp_path / "verify_expect_fp.json"
    cli_res = runner.invoke(
        cli,
        [
            "audit-verify",
            "-p",
            str(pack_res.zip_path),
            "--profile",
            "audit-grade",
            "--public-key-dir",
            str(keyring_dir),
        ],
    )
    assert cli_res.exit_code == 2
    assert "attestation key_id must match signature key_id in audit-grade mode" in cli_res.output


def test_audit_verify_expect_pack_fingerprint_mismatch(sample_run_dir: Path, tmp_path: Path) -> None:
    _write_finalize_artifacts(sample_run_dir)
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=True,
    )
    assert pack_res.zip_path is not None

    private_key, public_key = _write_keypair(tmp_path)
    sign_audit_pack(pack_res.zip_path, private_key_path=private_key, public_key_path=public_key)
    attest_audit_pack(pack_res.zip_path, private_key_path=private_key, public_key_path=public_key)

    mismatch = "0" * 64
    assert mismatch != pack_res.checksums_sha256
    result = verify_audit_pack(
        pack_res.zip_path,
        public_key=public_key,
        require_signature=True,
        require_public_key=True,
        require_attestation=True,
        expect_pack_fingerprint=mismatch,
    )
    assert result.ok is False
    assert result.expected_pack_fingerprint == mismatch
    assert result.expected_pack_fingerprint_match is False
    assert any(issue.kind == "fingerprint_error" for issue in result.issues)

    runner = CliRunner()
    out_json = tmp_path / "verify_expect_fp.json"
    cli_res = runner.invoke(
        cli,
        [
            "audit-verify",
            "-p",
            str(pack_res.zip_path),
            "--profile",
            "audit-grade",
            "--public-key",
            str(public_key),
            "--expect-pack-fingerprint",
            mismatch,
            "--json-out",
            str(out_json),
        ],
    )
    assert cli_res.exit_code == 2
    assert "Expected fingerprint match: False" in cli_res.output
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["expected_pack_fingerprint"] == mismatch
    assert payload["expected_pack_fingerprint_match"] is False
