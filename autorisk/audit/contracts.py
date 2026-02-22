"""Typed JSON contracts for audit documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from autorisk.audit._crypto import canonical_json_bytes

_ModelT = TypeVar("_ModelT", bound="_AuditJsonContract")


class _AuditJsonContract(BaseModel):
    """Shared helpers for canonical and pretty JSON persistence."""

    model_config = ConfigDict(extra="allow")

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    def to_json_text(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json_text(), encoding="utf-8")

    @classmethod
    def read_json(cls: type[_ModelT], path: Path) -> _ModelT:
        loaded = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must be a JSON object")
        return cls.model_validate(loaded)


class AuditSignatureSigned(_AuditJsonContract):
    checksums_sha256: str
    manifest_sha256: str
    generated_at_utc: str
    key_id: str
    key_label: str | None = None


class AuditSignatureDocument(_AuditJsonContract):
    schema_version: int = 1
    algorithm: str = "ed25519"
    signed: AuditSignatureSigned
    signature: str
    public_key_pem: str | None = None


class AuditAttestationSigned(_AuditJsonContract):
    generated_at_utc: str
    key_id: str
    pack_fingerprint: str
    finalize_record_sha256: str
    audit_validate_report_sha256: str
    key_label: str | None = None


class AuditAttestationDocument(_AuditJsonContract):
    schema_version: int = 1
    algorithm: str = "ed25519"
    signed: AuditAttestationSigned
    signature: str
    public_key_pem: str | None = None


class FinalizeRunRecord(_AuditJsonContract):
    schema_version: int = 1
    created_at_utc: str
    run_dir: str
    pack_dir: str
    zip_path: str = ""
    autorisk_version: str
    python_version: str
    platform: str
    pack_fingerprint: str
    signature_present: bool
    signature_verified: bool | None = None
    signature_key_id: str = ""
    signature_key_source: str = ""
    policy_source: dict[str, Any] = Field(default_factory=dict)
    policy_sha256: str = ""
    revocation_file: str = ""
    revocation_file_sha256: str = ""
    revoked_key_ids_count: int = 0
    verification_issues: int = 0
    enforce: bool
    audit_grade: bool
    require_signature: bool
    require_trusted_key: bool
    trust_embedded_public_key: bool
    verifier_bundle_path: str = ""
    validate_ok: bool
    validate_issues_count: int
    validate_report_path: str
    validate_report_sha256: str
    handoff_path: str = ""
    handoff_checksums_sha256: str = ""
    handoff_pack_zip_sha256: str = ""
    handoff_verifier_bundle_zip_sha256: str = ""
    handoff_anchor_checksums_sha256: str = ""
    handoff_anchor_verifier_bundle_zip_sha256: str = ""


def write_finalize_record(
    path: Path, payload: FinalizeRunRecord | dict[str, Any]
) -> FinalizeRunRecord:
    record = (
        payload
        if isinstance(payload, FinalizeRunRecord)
        else FinalizeRunRecord.model_validate(payload)
    )
    record.write_json(path)
    return record
