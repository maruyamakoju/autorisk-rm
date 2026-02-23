"""Service-layer orchestration for `audit-verify`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from autorisk.audit.verify import verify_audit_pack

EmitFn = Callable[[str, bool], None]


@dataclass(slots=True)
class AuditVerifyServiceRequest:
    pack: str
    profile: str
    strict: bool
    public_key: str | None
    public_key_dir: str | None
    require_signature: bool
    require_public_key: bool
    require_attestation: bool
    trust_embedded_public_key: bool
    revoked_key_ids: set[str]
    expect_pack_fingerprint: str | None
    json_out: str | None


def _format_effective_flags(flags: dict[str, object]) -> str:
    parts: list[str] = []
    for key, value in flags.items():
        if isinstance(value, bool):
            parts.append(f"{key}={'true' if value else 'false'}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def run_audit_verify(request: AuditVerifyServiceRequest, *, emit: EmitFn) -> None:
    """Execute `audit-verify` orchestration without Click-bound logic."""
    strict = bool(request.strict)
    require_signature = bool(request.require_signature)
    require_public_key = bool(request.require_public_key)
    require_attestation = bool(request.require_attestation)
    trust_embedded_public_key = bool(request.trust_embedded_public_key)
    require_attestation_key_match_signature = False

    profile_value = str(request.profile).strip().lower()
    if profile_value == "audit-grade":
        strict = True
        require_signature = True
        require_public_key = True
        require_attestation = True
        trust_embedded_public_key = False
        require_attestation_key_match_signature = True
        emit(
            "[audit-grade] "
            + _format_effective_flags(
                {
                    "strict": strict,
                    "require_signature": require_signature,
                    "require_public_key": require_public_key,
                    "require_attestation": require_attestation,
                    "trust_embedded_public_key": trust_embedded_public_key,
                    "require_attestation_key_match_signature": require_attestation_key_match_signature,
                }
            ),
            False,
        )
    elif profile_value == "default":
        emit(
            "[default] diagnostics mode: crypto requirements are NOT enforced. Do not use for audit-grade acceptance.",
            True,
        )
        emit(
            "Use: audit-verify --profile audit-grade --public-key-dir <TRUSTED_KEYRING> [--expect-pack-fingerprint <TICKET_FP>]",
            True,
        )

    result = verify_audit_pack(
        request.pack,
        strict=strict,
        public_key=request.public_key,
        public_key_dir=request.public_key_dir,
        require_signature=require_signature,
        require_public_key=require_public_key,
        require_attestation=require_attestation,
        require_attestation_key_match_signature=require_attestation_key_match_signature,
        expect_pack_fingerprint=request.expect_pack_fingerprint,
        trust_embedded_public_key=trust_embedded_public_key,
        revoked_key_ids=set(request.revoked_key_ids),
    )

    emit(f"Source: {result.source}", False)
    emit(f"Mode: {result.mode}", False)
    emit(f"Pack root: {result.pack_root}", False)
    emit(f"Checksums: {result.checksums_path}", False)
    emit(f"Checksums SHA256: {result.checksums_sha256}", False)
    emit(f"Pack fingerprint: {result.checksums_sha256}", False)
    emit(f"Expected files: {result.expected_files}", False)
    emit(f"Verified files: {result.verified_files}", False)
    emit(f"Signature present: {result.signature_present}", False)
    if result.signature_present:
        emit(f"Signature path: {result.signature_path}", False)
        emit(f"Signature key id: {result.signature_key_id}", False)
        emit(f"Signature key source: {result.signature_key_source or 'none'}", False)
        emit(f"Signature verified: {result.signature_verified}", False)
    emit(f"Attestation present: {result.attestation_present}", False)
    if result.attestation_present:
        emit(f"Attestation path: {result.attestation_path}", False)
        emit(f"Attestation key id: {result.attestation_key_id}", False)
        emit(
            f"Attestation key source: {result.attestation_key_source or 'none'}", False
        )
    emit(f"Attestation verified: {result.attestation_verified}", False)
    if result.expected_pack_fingerprint != "":
        emit(f"Expected fingerprint: {result.expected_pack_fingerprint}", False)
        emit(
            f"Expected fingerprint match: {result.expected_pack_fingerprint_match}",
            False,
        )
    unchecked_files = list(result.unchecked_files or [])
    emit(f"Unchecked files: {len(unchecked_files)}", False)
    for rel in unchecked_files[:20]:
        emit(f"  - {rel}", False)
    if len(unchecked_files) > 20:
        emit(f"  ... ({len(unchecked_files) - 20} more)", False)
    emit(f"Issues: {len(result.issues)}", False)

    if request.json_out is not None and str(request.json_out).strip() != "":
        out_path = Path(request.json_out).expanduser().resolve()
        out_path.write_text(result.to_json(), encoding="utf-8")
        emit(f"Wrote: {out_path}", False)

    if result.issues:
        for issue in result.issues[:50]:
            emit(f"- {issue.kind}: {issue.path} {issue.detail}".rstrip(), False)
        if len(result.issues) > 50:
            emit(f"... ({len(result.issues) - 50} more)", False)
        if strict:
            raise SystemExit(2)
