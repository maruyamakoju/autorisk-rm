"""Attestation for non-checksummed audit artifacts."""

from __future__ import annotations

import base64
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import ValidationError

from autorisk.audit._crypto import (
    assert_matching_keypair,
    load_private_key,
    load_public_key,
    public_key_id,
    public_key_pem,
    sha256_bytes,
    sha256_file,
    utc_now_iso,
)
from autorisk.audit.contracts import (
    AuditAttestationDocument,
    AuditAttestationSigned,
)
from autorisk.audit._pack_locator import resolve_pack_root_dir, resolve_pack_root_zip
from autorisk.audit._zip_utils import rewrite_zip_member

CHECKSUMS_FILENAME = "checksums.sha256.txt"
ATTESTATION_FILENAME = "attestation.json"
ATTESTATION_SCHEMA_VERSION = 1
ATTESTATION_ALGORITHM = "ed25519"
FINALIZE_RECORD_REL = "run_artifacts/finalize_record.json"
VALIDATE_REPORT_REL = "run_artifacts/audit_validate_report.json"


@dataclass
class AuditAttestationResult:
    source: Path
    mode: str  # "dir" | "zip"
    attestation_path: str
    key_id: str
    pack_fingerprint: str
    finalize_record_sha256: str
    audit_validate_report_sha256: str


@dataclass
class PackAttestationContext:
    source: Path
    mode: str
    attestation_path: str
    attestation_doc: dict[str, Any] | None
    pack_fingerprint: str
    finalize_record_sha256: str
    audit_validate_report_sha256: str


def _write_zip_member(zip_path: Path, member_name: str, payload: bytes) -> None:
    rewrite_zip_member(
        zip_path=zip_path,
        member_name=member_name,
        payload=payload,
        temp_prefix="autorisk-attest-",
    )


def _resolve_pack_root_dir(pack_dir: Path) -> tuple[Path, Path]:
    return resolve_pack_root_dir(pack_dir, checksums_filename=CHECKSUMS_FILENAME)


def _resolve_pack_root_zip(zip_path: Path) -> tuple[str, str]:
    return resolve_pack_root_zip(zip_path, checksums_filename=CHECKSUMS_FILENAME)


def _build_attestation_document(
    *,
    private_key: Ed25519PrivateKey,
    public_key: Ed25519PublicKey,
    pack_fingerprint: str,
    finalize_record_sha256: str,
    audit_validate_report_sha256: str,
    key_label: str | None = None,
    include_public_key: bool = False,
) -> AuditAttestationDocument:
    signed_payload = AuditAttestationSigned(
        generated_at_utc=utc_now_iso(),
        key_id=public_key_id(public_key),
        pack_fingerprint=pack_fingerprint,
        finalize_record_sha256=finalize_record_sha256,
        audit_validate_report_sha256=audit_validate_report_sha256,
    )
    if key_label is not None and str(key_label).strip() != "":
        signed_payload.key_label = str(key_label).strip()
    signature_bytes = private_key.sign(signed_payload.canonical_bytes())
    payload = AuditAttestationDocument(
        schema_version=ATTESTATION_SCHEMA_VERSION,
        algorithm=ATTESTATION_ALGORITHM,
        signed=signed_payload,
        signature=base64.b64encode(signature_bytes).decode("ascii"),
    )
    if include_public_key:
        payload.public_key_pem = public_key_pem(public_key)
    return payload


def attest_audit_pack(
    path: str | Path,
    *,
    private_key_path: str | Path,
    private_key_password: str | bytes | None = None,
    key_label: str | None = None,
    include_public_key: bool = False,
    public_key_path: str | Path | None = None,
) -> AuditAttestationResult:
    """Sign attestation over non-checksummed run artifacts."""
    source_path = Path(path).expanduser().resolve()
    private_key = load_private_key(
        Path(private_key_path).expanduser().resolve(),
        password=private_key_password,
    )
    if public_key_path is not None and str(public_key_path).strip() != "":
        public_key = load_public_key(Path(public_key_path).expanduser().resolve())
        assert_matching_keypair(
            private_key=private_key,
            public_key=public_key,
            context="attest_audit_pack",
        )
    else:
        public_key = private_key.public_key()

    if source_path.is_dir():
        pack_root, checksums_path = _resolve_pack_root_dir(source_path)
        finalize_record_path = pack_root / FINALIZE_RECORD_REL
        validate_report_path = pack_root / VALIDATE_REPORT_REL
        if not finalize_record_path.exists() or not finalize_record_path.is_file():
            raise FileNotFoundError(f"missing {FINALIZE_RECORD_REL} under: {pack_root}")
        if not validate_report_path.exists() or not validate_report_path.is_file():
            raise FileNotFoundError(f"missing {VALIDATE_REPORT_REL} under: {pack_root}")

        pack_fingerprint = sha256_file(checksums_path)
        finalize_record_sha256 = sha256_file(finalize_record_path)
        audit_validate_report_sha256 = sha256_file(validate_report_path)
        attestation_doc = _build_attestation_document(
            private_key=private_key,
            public_key=public_key,
            pack_fingerprint=pack_fingerprint,
            finalize_record_sha256=finalize_record_sha256,
            audit_validate_report_sha256=audit_validate_report_sha256,
            key_label=key_label,
            include_public_key=include_public_key,
        )
        attestation_path = pack_root / ATTESTATION_FILENAME
        attestation_doc.write_json(attestation_path)
        return AuditAttestationResult(
            source=source_path,
            mode="dir",
            attestation_path=str(attestation_path),
            key_id=str(attestation_doc.signed.key_id),
            pack_fingerprint=pack_fingerprint,
            finalize_record_sha256=finalize_record_sha256,
            audit_validate_report_sha256=audit_validate_report_sha256,
        )

    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        prefix, checksums_name = _resolve_pack_root_zip(source_path)
        root_prefix = f"{prefix}/" if prefix else ""
        finalize_name = (
            f"{root_prefix}{FINALIZE_RECORD_REL}"
            if root_prefix
            else FINALIZE_RECORD_REL
        )
        validate_name = (
            f"{root_prefix}{VALIDATE_REPORT_REL}"
            if root_prefix
            else VALIDATE_REPORT_REL
        )
        attestation_name = (
            f"{root_prefix}{ATTESTATION_FILENAME}"
            if root_prefix
            else ATTESTATION_FILENAME
        )

        with zipfile.ZipFile(source_path, "r") as zf:
            member_set = set(zf.namelist())
            if finalize_name not in member_set:
                raise FileNotFoundError(
                    f"missing {FINALIZE_RECORD_REL} inside zip: {source_path}"
                )
            if validate_name not in member_set:
                raise FileNotFoundError(
                    f"missing {VALIDATE_REPORT_REL} inside zip: {source_path}"
                )

            pack_fingerprint = sha256_bytes(zf.read(checksums_name))
            finalize_record_sha256 = sha256_bytes(zf.read(finalize_name))
            audit_validate_report_sha256 = sha256_bytes(zf.read(validate_name))

        attestation_doc = _build_attestation_document(
            private_key=private_key,
            public_key=public_key,
            pack_fingerprint=pack_fingerprint,
            finalize_record_sha256=finalize_record_sha256,
            audit_validate_report_sha256=audit_validate_report_sha256,
            key_label=key_label,
            include_public_key=include_public_key,
        )
        _write_zip_member(
            source_path,
            attestation_name,
            attestation_doc.to_json_text().encode("utf-8"),
        )
        return AuditAttestationResult(
            source=source_path,
            mode="zip",
            attestation_path=f"{source_path}!{attestation_name}",
            key_id=str(attestation_doc.signed.key_id),
            pack_fingerprint=pack_fingerprint,
            finalize_record_sha256=finalize_record_sha256,
            audit_validate_report_sha256=audit_validate_report_sha256,
        )

    raise FileNotFoundError(f"pack path not found or unsupported: {source_path}")


def load_pack_attestation_context(path: str | Path) -> PackAttestationContext:
    """Load attestation document and expected hash values from a pack."""
    source_path = Path(path).expanduser().resolve()

    if source_path.is_dir():
        pack_root, checksums_path = _resolve_pack_root_dir(source_path)
        finalize_record_path = pack_root / FINALIZE_RECORD_REL
        validate_report_path = pack_root / VALIDATE_REPORT_REL
        attestation_path = pack_root / ATTESTATION_FILENAME
        if not finalize_record_path.exists() or not finalize_record_path.is_file():
            raise FileNotFoundError(f"missing {FINALIZE_RECORD_REL} under: {pack_root}")
        if not validate_report_path.exists() or not validate_report_path.is_file():
            raise FileNotFoundError(f"missing {VALIDATE_REPORT_REL} under: {pack_root}")

        attestation_doc: dict[str, Any] | None = None
        if attestation_path.exists() and attestation_path.is_file():
            loaded = json.loads(
                attestation_path.read_text(encoding="utf-8", errors="replace")
            )
            if isinstance(loaded, dict):
                attestation_doc = AuditAttestationDocument.model_validate(
                    loaded
                ).as_dict()
            else:
                raise ValueError("attestation.json must be an object")
        return PackAttestationContext(
            source=source_path,
            mode="dir",
            attestation_path=str(attestation_path),
            attestation_doc=attestation_doc,
            pack_fingerprint=sha256_file(checksums_path),
            finalize_record_sha256=sha256_file(finalize_record_path),
            audit_validate_report_sha256=sha256_file(validate_report_path),
        )

    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        prefix, checksums_name = _resolve_pack_root_zip(source_path)
        root_prefix = f"{prefix}/" if prefix else ""
        finalize_name = (
            f"{root_prefix}{FINALIZE_RECORD_REL}"
            if root_prefix
            else FINALIZE_RECORD_REL
        )
        validate_name = (
            f"{root_prefix}{VALIDATE_REPORT_REL}"
            if root_prefix
            else VALIDATE_REPORT_REL
        )
        attestation_name = (
            f"{root_prefix}{ATTESTATION_FILENAME}"
            if root_prefix
            else ATTESTATION_FILENAME
        )
        with zipfile.ZipFile(source_path, "r") as zf:
            member_set = set(zf.namelist())
            if finalize_name not in member_set:
                raise FileNotFoundError(
                    f"missing {FINALIZE_RECORD_REL} inside zip: {source_path}"
                )
            if validate_name not in member_set:
                raise FileNotFoundError(
                    f"missing {VALIDATE_REPORT_REL} inside zip: {source_path}"
                )

            attestation_doc = None
            if attestation_name in member_set:
                loaded = json.loads(
                    zf.read(attestation_name).decode("utf-8", errors="replace")
                )
                if isinstance(loaded, dict):
                    attestation_doc = AuditAttestationDocument.model_validate(
                        loaded
                    ).as_dict()
                else:
                    raise ValueError("attestation.json must be an object")
            return PackAttestationContext(
                source=source_path,
                mode="zip",
                attestation_path=f"{source_path}!{attestation_name}",
                attestation_doc=attestation_doc,
                pack_fingerprint=sha256_bytes(zf.read(checksums_name)),
                finalize_record_sha256=sha256_bytes(zf.read(finalize_name)),
                audit_validate_report_sha256=sha256_bytes(zf.read(validate_name)),
            )

    raise FileNotFoundError(f"pack path not found or unsupported: {source_path}")


def verify_attestation_document(
    attestation_doc: dict[str, Any],
    *,
    pack_fingerprint: str,
    finalize_record_sha256: str,
    audit_validate_report_sha256: str,
    public_key: Ed25519PublicKey,
) -> tuple[bool, str]:
    """Verify attestation payload and signed hashes."""
    if not isinstance(attestation_doc, dict):
        return False, "attestation.json must be an object"
    try:
        parsed = AuditAttestationDocument.model_validate(attestation_doc)
    except ValidationError as exc:
        return (
            False,
            f"invalid attestation document: {exc.errors()[0].get('msg', 'validation error')}",
        )
    if int(parsed.schema_version) != ATTESTATION_SCHEMA_VERSION:
        return False, f"unsupported attestation schema_version: {parsed.schema_version}"
    if str(parsed.algorithm).lower() != ATTESTATION_ALGORITHM:
        return False, f"unsupported algorithm: {parsed.algorithm}"

    signed = parsed.signed
    expected_pack = str(signed.pack_fingerprint).strip().lower()
    expected_finalize = str(signed.finalize_record_sha256).strip().lower()
    expected_validate = str(signed.audit_validate_report_sha256).strip().lower()
    if expected_pack != pack_fingerprint.lower():
        return False, "signed pack_fingerprint does not match checksums hash"
    if expected_finalize != finalize_record_sha256.lower():
        return (
            False,
            "signed finalize_record_sha256 does not match run_artifacts/finalize_record.json",
        )
    if expected_validate != audit_validate_report_sha256.lower():
        return (
            False,
            "signed audit_validate_report_sha256 does not match run_artifacts/audit_validate_report.json",
        )

    sig_text = str(parsed.signature).strip()
    if sig_text == "":
        return False, "missing signature"
    try:
        sig_bytes = base64.b64decode(sig_text, validate=True)
    except Exception as exc:
        return False, f"invalid base64 signature: {exc}"

    try:
        public_key.verify(sig_bytes, signed.canonical_bytes())
    except InvalidSignature:
        return False, "invalid signature"
    except Exception as exc:
        return False, f"signature verification failed: {exc}"

    expected_key_id = str(signed.key_id).strip()
    if expected_key_id == "":
        return False, "missing key_id in signed payload"
    if expected_key_id != public_key_id(public_key):
        return False, "key_id does not match verification key"

    return True, ""
