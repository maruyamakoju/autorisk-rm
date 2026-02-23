from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from omegaconf import OmegaConf

from autorisk.audit.attestation import attest_audit_pack
from autorisk.audit.pack import build_audit_pack
from autorisk.audit.sign import sign_audit_pack
from autorisk.audit.verify import verify_audit_pack
from autorisk.cli import cli


def _write_ed25519_keypair(
    tmp_path: Path,
    *,
    password: str | None = None,
    prefix: str = "",
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_path = tmp_path / f"{prefix}private_key.pem"
    public_path = tmp_path / f"{prefix}public_key.pem"
    encryption = (
        serialization.NoEncryption()
        if password is None
        else serialization.BestAvailableEncryption(password.encode("utf-8"))
    )
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption,
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
                "signature_present": False,
                "signature_key_id": "",
                "validate_ok": True,
                "validate_issues_count": 0,
                "validate_report_path": "run_artifacts/audit_validate_report.json",
                "validate_report_sha256": "",
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


def test_audit_sign_and_verify_for_directory(
    sample_run_dir: Path, tmp_path: Path
) -> None:
    private_key, public_key = _write_ed25519_keypair(tmp_path)
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=False,
    )

    sign_res = sign_audit_pack(
        pack_res.output_dir,
        private_key_path=private_key,
        public_key_path=public_key,
    )
    assert sign_res.signature_path.endswith("signature.json")

    verify_ok = verify_audit_pack(pack_res.output_dir, public_key=public_key)
    assert verify_ok.ok is True
    assert verify_ok.signature_present is True
    assert verify_ok.signature_verified is True

    checksums_path = Path(verify_ok.checksums_path)
    checksums_path.write_text(
        checksums_path.read_text(encoding="utf-8") + "# tamper\n",
        encoding="utf-8",
    )
    verify_ng = verify_audit_pack(pack_res.output_dir, public_key=public_key)
    assert verify_ng.ok is False
    assert any(i.kind == "signature_error" for i in verify_ng.issues)


def test_audit_sign_and_verify_for_zip(sample_run_dir: Path, tmp_path: Path) -> None:
    private_key, public_key = _write_ed25519_keypair(tmp_path)
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=True,
    )
    assert pack_res.zip_path is not None

    sign_res = sign_audit_pack(
        pack_res.zip_path,
        private_key_path=private_key,
        public_key_path=public_key,
        include_public_key=True,
    )
    assert sign_res.mode == "zip"

    verify_no_trust = verify_audit_pack(pack_res.zip_path)
    assert verify_no_trust.ok is True
    assert verify_no_trust.signature_present is True
    assert verify_no_trust.signature_verified is None

    verify_embedded = verify_audit_pack(
        pack_res.zip_path, trust_embedded_public_key=True
    )
    assert verify_embedded.ok is True
    assert verify_embedded.signature_present is True
    assert verify_embedded.signature_verified is True
    assert verify_embedded.signature_key_source == "embedded"

    verify_ok = verify_audit_pack(
        pack_res.zip_path,
        public_key=public_key,
        require_signature=True,
        require_public_key=True,
    )
    assert verify_ok.ok is True
    assert verify_ok.signature_present is True
    assert verify_ok.signature_verified is True


def test_audit_verify_uses_keyring_directory(
    sample_run_dir: Path, tmp_path: Path
) -> None:
    private_key, public_key = _write_ed25519_keypair(tmp_path, prefix="active_")
    _, old_public_key = _write_ed25519_keypair(tmp_path, prefix="old_")
    keyring_dir = tmp_path / "trusted_keys"
    keyring_dir.mkdir(parents=True, exist_ok=True)
    (keyring_dir / "active.pem").write_bytes(public_key.read_bytes())
    (keyring_dir / "old.pem").write_bytes(old_public_key.read_bytes())

    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=False,
    )
    sign_res = sign_audit_pack(
        pack_res.output_dir,
        private_key_path=private_key,
        public_key_path=public_key,
    )
    verify_res = verify_audit_pack(
        pack_res.output_dir,
        public_key_dir=keyring_dir,
        require_signature=True,
        require_public_key=True,
    )
    assert verify_res.ok is True
    assert verify_res.signature_verified is True
    assert verify_res.signature_key_id == sign_res.key_id
    assert verify_res.signature_key_source.endswith("active.pem")


def test_audit_verify_fails_for_revoked_key(
    sample_run_dir: Path, tmp_path: Path
) -> None:
    private_key, public_key = _write_ed25519_keypair(tmp_path)
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=False,
    )
    sign_res = sign_audit_pack(
        pack_res.output_dir,
        private_key_path=private_key,
        public_key_path=public_key,
    )
    verify_res = verify_audit_pack(
        pack_res.output_dir,
        public_key=public_key,
        require_signature=True,
        require_public_key=True,
        revoked_key_ids={sign_res.key_id},
    )
    assert verify_res.ok is False
    assert any(
        i.kind == "signature_error" and "revoked" in i.detail for i in verify_res.issues
    )


def test_audit_sign_supports_encrypted_private_key(
    sample_run_dir: Path, tmp_path: Path
) -> None:
    password = "very-secret-passphrase"
    private_key, public_key = _write_ed25519_keypair(tmp_path, password=password)
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=False,
    )
    sign_audit_pack(
        pack_res.output_dir,
        private_key_path=private_key,
        private_key_password=password,
        public_key_path=public_key,
    )
    verify_res = verify_audit_pack(
        pack_res.output_dir,
        public_key=public_key,
        require_signature=True,
        require_public_key=True,
    )
    assert verify_res.ok is True
    assert verify_res.signature_verified is True


def test_audit_sign_rejects_mismatched_public_key(
    sample_run_dir: Path, tmp_path: Path
) -> None:
    private_key, _ = _write_ed25519_keypair(tmp_path / "signing")
    _, wrong_public_key = _write_ed25519_keypair(tmp_path / "wrong")
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=False,
    )

    with pytest.raises(ValueError, match="does not match the private key"):
        sign_audit_pack(
            pack_res.output_dir,
            private_key_path=private_key,
            public_key_path=wrong_public_key,
        )


def test_audit_attest_rejects_mismatched_public_key(
    sample_run_dir: Path, tmp_path: Path
) -> None:
    _write_finalize_artifacts(sample_run_dir)
    private_key, _ = _write_ed25519_keypair(tmp_path / "signing")
    _, wrong_public_key = _write_ed25519_keypair(tmp_path / "wrong")
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=True,
    )
    assert pack_res.zip_path is not None

    with pytest.raises(ValueError, match="does not match the private key"):
        attest_audit_pack(
            pack_res.zip_path,
            private_key_path=private_key,
            public_key_path=wrong_public_key,
        )


def test_audit_verify_fails_with_wrong_public_key(
    sample_run_dir: Path, tmp_path: Path
) -> None:
    private_key, public_key = _write_ed25519_keypair(tmp_path)
    _, wrong_public_key = _write_ed25519_keypair(tmp_path / "wrong")
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=False,
    )

    sign_audit_pack(
        pack_res.output_dir,
        private_key_path=private_key,
        public_key_path=public_key,
    )
    verify_wrong = verify_audit_pack(
        pack_res.output_dir,
        public_key=wrong_public_key,
        require_signature=True,
        require_public_key=True,
    )
    assert verify_wrong.ok is False
    assert any(i.kind == "signature_error" for i in verify_wrong.issues)


def test_audit_verify_fails_on_invalid_signature_json(
    sample_run_dir: Path, tmp_path: Path
) -> None:
    private_key, public_key = _write_ed25519_keypair(tmp_path)
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=False,
    )
    sign_audit_pack(
        pack_res.output_dir,
        private_key_path=private_key,
        public_key_path=public_key,
    )

    signature_path = pack_res.output_dir / "signature.json"
    signature_obj = json.loads(signature_path.read_text(encoding="utf-8"))
    signature_obj["signature"] = "!!!not-base64!!!"
    signature_path.write_text(
        json.dumps(signature_obj, ensure_ascii=False), encoding="utf-8"
    )

    verify_res = verify_audit_pack(
        pack_res.output_dir,
        public_key=public_key,
        require_signature=True,
        require_public_key=True,
    )
    assert verify_res.ok is False
    assert any(i.kind == "signature_error" for i in verify_res.issues)


def test_audit_verify_require_signature_and_public_key_flags(
    sample_run_dir: Path,
) -> None:
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=False,
    )

    missing_signature = verify_audit_pack(pack_res.output_dir, require_signature=True)
    assert missing_signature.ok is False
    assert any(i.kind == "signature_error" for i in missing_signature.issues)

    require_pub_without_key = verify_audit_pack(
        pack_res.output_dir,
        require_signature=False,
        require_public_key=True,
    )
    assert require_pub_without_key.ok is False
    assert any(i.kind == "signature_error" for i in require_pub_without_key.issues)


def test_audit_attest_cli_generates_verifiable_attestation(
    sample_run_dir: Path, tmp_path: Path
) -> None:
    _write_finalize_artifacts(sample_run_dir)
    private_key, public_key = _write_ed25519_keypair(tmp_path)
    pack_res = build_audit_pack(
        run_dir=sample_run_dir,
        cfg=_sample_cfg(),
        include_clips=False,
        create_zip=True,
    )
    assert pack_res.zip_path is not None

    runner = CliRunner()
    cli_res = runner.invoke(
        cli,
        [
            "audit-attest",
            "-p",
            str(pack_res.zip_path),
            "--private-key",
            str(private_key),
            "--public-key",
            str(public_key),
        ],
        catch_exceptions=False,
    )
    assert cli_res.exit_code == 0, cli_res.output
    assert "Attestation:" in cli_res.output

    sign_audit_pack(
        pack_res.zip_path,
        private_key_path=private_key,
        public_key_path=public_key,
    )
    verify_res = verify_audit_pack(
        pack_res.zip_path,
        public_key=public_key,
        require_signature=True,
        require_public_key=True,
        require_attestation=True,
    )
    assert verify_res.ok is True
    assert verify_res.attestation_present is True
    assert verify_res.attestation_verified is True
