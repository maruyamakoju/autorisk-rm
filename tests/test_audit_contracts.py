from __future__ import annotations

from pathlib import Path

from autorisk.audit.contracts import (
    AuditSignatureDocument,
    AuditSignatureSigned,
    FinalizeRunRecord,
    write_finalize_record,
)


def _finalize_payload(tmp_path: Path) -> dict[str, object]:
    run_dir = tmp_path / "run"
    pack_dir = tmp_path / "pack"
    return {
        "created_at_utc": "2026-02-22T00:00:00+00:00",
        "run_dir": str(run_dir),
        "pack_dir": str(pack_dir),
        "zip_path": str(pack_dir / "pack.zip"),
        "autorisk_version": "0.1.0",
        "python_version": "3.10",
        "platform": "test",
        "pack_fingerprint": "a" * 64,
        "signature_present": True,
        "signature_verified": True,
        "signature_key_id": "test-key-id",
        "signature_key_source": "trusted.pem",
        "policy_source": {"source_type": "test"},
        "policy_sha256": "b" * 64,
        "revocation_file": "",
        "revocation_file_sha256": "",
        "revoked_key_ids_count": 0,
        "verification_issues": 0,
        "enforce": True,
        "audit_grade": True,
        "require_signature": True,
        "require_trusted_key": True,
        "trust_embedded_public_key": False,
        "verifier_bundle_path": "",
        "validate_ok": True,
        "validate_issues_count": 0,
        "validate_report_path": str(run_dir / "audit_validate_report.json"),
        "validate_report_sha256": "c" * 64,
        "handoff_path": "",
        "handoff_checksums_sha256": "",
        "handoff_pack_zip_sha256": "",
        "handoff_verifier_bundle_zip_sha256": "",
        "handoff_anchor_checksums_sha256": "d" * 64,
        "handoff_anchor_verifier_bundle_zip_sha256": "e" * 64,
    }


def test_signature_contract_roundtrip(tmp_path: Path) -> None:
    doc = AuditSignatureDocument(
        signed=AuditSignatureSigned(
            checksums_sha256="1" * 64,
            manifest_sha256="2" * 64,
            generated_at_utc="2026-02-22T00:00:00+00:00",
            key_id="key-1",
        ),
        signature="YWJj",
    )
    out = tmp_path / "signature.json"
    doc.write_json(out)

    loaded = AuditSignatureDocument.read_json(out)
    assert loaded.schema_version == 1
    assert loaded.as_dict() == doc.as_dict()
    assert loaded.canonical_bytes() == doc.canonical_bytes()


def test_finalize_record_contract_write_helpers(tmp_path: Path) -> None:
    out_from_dict = tmp_path / "finalize_dict.json"
    model = write_finalize_record(out_from_dict, _finalize_payload(tmp_path))
    assert isinstance(model, FinalizeRunRecord)
    assert model.schema_version == 1

    out_from_model = tmp_path / "finalize_model.json"
    write_finalize_record(out_from_model, model)
    loaded = FinalizeRunRecord.read_json(out_from_model)
    assert loaded.pack_fingerprint == "a" * 64
    assert loaded.validate_ok is True
