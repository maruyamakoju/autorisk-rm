"""Sign audit packs and verify signature payloads."""

from __future__ import annotations

import base64
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
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
    AuditSignatureDocument,
    AuditSignatureSigned,
)
from autorisk.audit._pack_locator import resolve_pack_root_dir, resolve_pack_root_zip
from autorisk.audit._zip_utils import rewrite_zip_member
from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)

CHECKSUMS_FILENAME = "checksums.sha256.txt"
MANIFEST_FILENAME = "manifest.json"
SIGNATURE_FILENAME = "signature.json"
SIGNATURE_SCHEMA_VERSION = 1
SIGNATURE_ALGORITHM = "ed25519"


@dataclass
class AuditSignResult:
    source: Path
    mode: str  # "dir" | "zip"
    signature_path: str
    key_id: str
    checksums_sha256: str
    manifest_sha256: str


def _signed_key_id(document: dict[str, Any], *, key_id_hint: str | None = None) -> str:
    if key_id_hint is not None and str(key_id_hint).strip() != "":
        return str(key_id_hint).strip()
    if not isinstance(document, dict):
        return ""
    signed = document.get("signed")
    if isinstance(signed, dict):
        key_id = str(signed.get("key_id", "")).strip()
        if key_id != "":
            return key_id
    try:
        parsed = AuditSignatureDocument.model_validate(document)
    except ValidationError:
        return ""
    return str(parsed.signed.key_id).strip()


def _resolve_pack_root_dir(pack_dir: Path) -> tuple[Path, Path, Path]:
    pack_root, checksums_path = resolve_pack_root_dir(
        pack_dir,
        checksums_filename=CHECKSUMS_FILENAME,
    )
    manifest_path = pack_root / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {MANIFEST_FILENAME} under: {pack_root}")
    return pack_root, checksums_path, manifest_path


def _resolve_pack_root_zip(zip_path: Path) -> tuple[str, str, str]:
    prefix, checksums_name = resolve_pack_root_zip(
        zip_path,
        checksums_filename=CHECKSUMS_FILENAME,
    )
    with zipfile.ZipFile(zip_path, "r") as zf:
        manifest_name = f"{prefix}/{MANIFEST_FILENAME}" if prefix else MANIFEST_FILENAME
        if manifest_name not in set(zf.namelist()):
            raise FileNotFoundError(
                f"missing {MANIFEST_FILENAME} inside zip: {zip_path}"
            )
        return prefix, checksums_name, manifest_name


def _build_signature_document(
    *,
    private_key: Ed25519PrivateKey,
    public_key: Ed25519PublicKey,
    checksums_sha256: str,
    manifest_sha256: str,
    key_label: str | None = None,
    include_public_key: bool = False,
) -> AuditSignatureDocument:
    signed_payload = AuditSignatureSigned(
        checksums_sha256=checksums_sha256,
        manifest_sha256=manifest_sha256,
        generated_at_utc=utc_now_iso(),
        key_id=public_key_id(public_key),
    )
    if key_label is not None and str(key_label).strip() != "":
        signed_payload.key_label = str(key_label).strip()
    signature_bytes = private_key.sign(signed_payload.canonical_bytes())
    payload = AuditSignatureDocument(
        schema_version=SIGNATURE_SCHEMA_VERSION,
        algorithm=SIGNATURE_ALGORITHM,
        signed=signed_payload,
        signature=base64.b64encode(signature_bytes).decode("ascii"),
    )
    if include_public_key:
        payload.public_key_pem = public_key_pem(public_key)
    return payload


def _write_zip_member(zip_path: Path, member_name: str, payload: bytes) -> None:
    rewrite_zip_member(
        zip_path=zip_path,
        member_name=member_name,
        payload=payload,
        temp_prefix="autorisk-sign-",
    )


def sign_audit_pack(
    path: str | Path,
    *,
    private_key_path: str | Path,
    private_key_password: str | bytes | None = None,
    key_label: str | None = None,
    include_public_key: bool = False,
    public_key_path: str | Path | None = None,
) -> AuditSignResult:
    """Sign an audit pack directory or zip bundle using Ed25519."""
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
            context="sign_audit_pack",
        )
    else:
        public_key = private_key.public_key()

    if source_path.is_dir():
        pack_root, checksums_path, manifest_path = _resolve_pack_root_dir(source_path)
        checksums_sha256 = sha256_file(checksums_path)
        manifest_sha256 = sha256_file(manifest_path)
        signature_doc = _build_signature_document(
            private_key=private_key,
            public_key=public_key,
            checksums_sha256=checksums_sha256,
            manifest_sha256=manifest_sha256,
            key_label=key_label,
            include_public_key=include_public_key,
        )
        signature_path = pack_root / SIGNATURE_FILENAME
        signature_doc.write_json(signature_path)
        return AuditSignResult(
            source=source_path,
            mode="dir",
            signature_path=str(signature_path),
            key_id=str(signature_doc.signed.key_id),
            checksums_sha256=checksums_sha256,
            manifest_sha256=manifest_sha256,
        )

    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        prefix, checksums_name, manifest_name = _resolve_pack_root_zip(source_path)
        with zipfile.ZipFile(source_path, "r") as zf:
            checksums_sha256 = sha256_bytes(zf.read(checksums_name))
            manifest_sha256 = sha256_bytes(zf.read(manifest_name))
        signature_doc = _build_signature_document(
            private_key=private_key,
            public_key=public_key,
            checksums_sha256=checksums_sha256,
            manifest_sha256=manifest_sha256,
            key_label=key_label,
            include_public_key=include_public_key,
        )
        signature_name = (
            f"{prefix}/{SIGNATURE_FILENAME}" if prefix else SIGNATURE_FILENAME
        )
        _write_zip_member(
            source_path,
            signature_name,
            signature_doc.to_json_text().encode("utf-8"),
        )
        return AuditSignResult(
            source=source_path,
            mode="zip",
            signature_path=f"{source_path}!{signature_name}",
            key_id=str(signature_doc.signed.key_id),
            checksums_sha256=checksums_sha256,
            manifest_sha256=manifest_sha256,
        )

    raise FileNotFoundError(f"pack path not found or unsupported: {source_path}")


def resolve_public_key_for_verification(
    *,
    public_key_path: str | Path | None,
    public_key_dir: str | Path | None,
    signature_doc: dict[str, Any],
    key_id_hint: str | None = None,
    trust_embedded_public_key: bool = False,
) -> tuple[Ed25519PublicKey | None, str]:
    """Resolve verification key from CLI path or embedded key in signature.json."""
    if public_key_path is not None and str(public_key_path).strip() != "":
        path = Path(public_key_path).expanduser().resolve()
        return load_public_key(path), str(path)

    if public_key_dir is not None and str(public_key_dir).strip() != "":
        directory = Path(public_key_dir).expanduser().resolve()
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"public key directory not found: {directory}")

        key_id = _signed_key_id(signature_doc, key_id_hint=key_id_hint).lower()
        if key_id == "":
            raise ValueError("signature payload does not include key_id")

        pem_paths = sorted(p for p in directory.rglob("*.pem") if p.is_file())
        if not pem_paths:
            raise FileNotFoundError(
                f"no .pem files found in public key directory: {directory}"
            )

        for pem_path in pem_paths:
            try:
                key = load_public_key(pem_path)
            except Exception:
                continue
            if public_key_id(key).lower() == key_id:
                return key, str(pem_path)
        raise ValueError(f"no key in {directory} matches signature key_id={key_id}")

    if not trust_embedded_public_key:
        return None, ""

    embedded = signature_doc.get("public_key_pem")
    if isinstance(embedded, str) and embedded.strip() != "":
        key = serialization.load_pem_public_key(embedded.encode("utf-8"))
        if not isinstance(key, Ed25519PublicKey):
            raise ValueError("embedded public_key_pem is not Ed25519")
        return key, "embedded"

    return None, ""


def verify_signature_document(
    signature_doc: dict[str, Any],
    *,
    checksums_sha256: str,
    manifest_sha256: str,
    public_key: Ed25519PublicKey,
) -> tuple[bool, str]:
    """Verify signature payload and content hashes."""
    if not isinstance(signature_doc, dict):
        return False, "signature.json must be an object"
    try:
        parsed = AuditSignatureDocument.model_validate(signature_doc)
    except ValidationError as exc:
        return (
            False,
            f"invalid signature document: {exc.errors()[0].get('msg', 'validation error')}",
        )
    if str(parsed.algorithm).lower() != SIGNATURE_ALGORITHM:
        return False, f"unsupported algorithm: {parsed.algorithm}"

    signed = parsed.signed
    expected_checksums = str(signed.checksums_sha256).strip().lower()
    expected_manifest = str(signed.manifest_sha256).strip().lower()
    if expected_checksums != checksums_sha256.lower():
        return (
            False,
            "signed checksums_sha256 does not match actual checksums file hash",
        )
    if expected_manifest != manifest_sha256.lower():
        return False, "signed manifest_sha256 does not match actual manifest hash"

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
