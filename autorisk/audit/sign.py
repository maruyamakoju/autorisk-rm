"""Sign audit packs and verify signature payloads."""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_private_key(path: Path, *, password: str | bytes | None = None) -> Ed25519PrivateKey:
    key_bytes = path.read_bytes()
    password_bytes: bytes | None
    if password is None:
        password_bytes = None
    elif isinstance(password, bytes):
        password_bytes = password
    else:
        password_bytes = str(password).encode("utf-8")
    key = serialization.load_pem_private_key(key_bytes, password=password_bytes)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(f"private key is not Ed25519: {path}")
    return key


def _load_public_key(path: Path) -> Ed25519PublicKey:
    key_bytes = path.read_bytes()
    key = serialization.load_pem_public_key(key_bytes)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError(f"public key is not Ed25519: {path}")
    return key


def _public_key_pem(public_key: Ed25519PublicKey) -> str:
    return (
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
        .strip()
    )


def _key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()[:16]


def _signed_key_id(signature_doc: dict[str, Any]) -> str:
    signed = signature_doc.get("signed", {})
    if not isinstance(signed, dict):
        return ""
    return str(signed.get("key_id", "")).strip()


def _resolve_pack_root_dir(pack_dir: Path) -> tuple[Path, Path, Path]:
    candidates = sorted(pack_dir.rglob(CHECKSUMS_FILENAME), key=lambda p: (len(p.parts), str(p)))
    if not candidates:
        raise FileNotFoundError(f"missing {CHECKSUMS_FILENAME} under: {pack_dir}")
    checksums_path = candidates[0]
    pack_root = checksums_path.parent
    manifest_path = pack_root / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {MANIFEST_FILENAME} under: {pack_root}")
    return pack_root, checksums_path, manifest_path


def _resolve_pack_root_zip(zip_path: Path) -> tuple[str, str, str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [n for n in zf.namelist() if n.endswith(CHECKSUMS_FILENAME) and not n.endswith("/")]
        if not members:
            raise FileNotFoundError(f"missing {CHECKSUMS_FILENAME} inside zip: {zip_path}")
        checksums_name = min(members, key=lambda n: (n.count("/"), len(n)))
        prefix = checksums_name[: -len(CHECKSUMS_FILENAME)].rstrip("/")
        manifest_name = f"{prefix}/{MANIFEST_FILENAME}" if prefix else MANIFEST_FILENAME
        if manifest_name not in set(zf.namelist()):
            raise FileNotFoundError(f"missing {MANIFEST_FILENAME} inside zip: {zip_path}")
        return prefix, checksums_name, manifest_name


def _build_signature_document(
    *,
    private_key: Ed25519PrivateKey,
    public_key: Ed25519PublicKey,
    checksums_sha256: str,
    manifest_sha256: str,
    key_label: str | None = None,
    include_public_key: bool = False,
) -> dict[str, Any]:
    signed = {
        "checksums_sha256": checksums_sha256,
        "manifest_sha256": manifest_sha256,
        "generated_at_utc": _utc_now_iso(),
        "key_id": _key_id(public_key),
    }
    if key_label is not None and str(key_label).strip() != "":
        signed["key_label"] = str(key_label).strip()
    signature_bytes = private_key.sign(_canonical_json_bytes(signed))
    payload: dict[str, Any] = {
        "schema_version": SIGNATURE_SCHEMA_VERSION,
        "algorithm": SIGNATURE_ALGORITHM,
        "signed": signed,
        "signature": base64.b64encode(signature_bytes).decode("ascii"),
    }
    if include_public_key:
        payload["public_key_pem"] = _public_key_pem(public_key)
    return payload


def _write_zip_member(zip_path: Path, member_name: str, payload: bytes) -> None:
    with tempfile.TemporaryDirectory(prefix="autorisk-sign-") as tmp_dir:
        tmp_zip = Path(tmp_dir) / zip_path.name
        with zipfile.ZipFile(zip_path, "r") as src, zipfile.ZipFile(
            tmp_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as dst:
            for info in src.infolist():
                if info.filename == member_name:
                    continue
                if info.is_dir():
                    dst.writestr(info, b"")
                    continue
                dst.writestr(info, src.read(info.filename))
            dst.writestr(member_name, payload)
        tmp_zip.replace(zip_path)


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
    private_key = _load_private_key(
        Path(private_key_path).expanduser().resolve(),
        password=private_key_password,
    )
    if public_key_path is not None and str(public_key_path).strip() != "":
        public_key = _load_public_key(Path(public_key_path).expanduser().resolve())
    else:
        public_key = private_key.public_key()

    if source_path.is_dir():
        pack_root, checksums_path, manifest_path = _resolve_pack_root_dir(source_path)
        checksums_sha256 = _sha256_file(checksums_path)
        manifest_sha256 = _sha256_file(manifest_path)
        signature_doc = _build_signature_document(
            private_key=private_key,
            public_key=public_key,
            checksums_sha256=checksums_sha256,
            manifest_sha256=manifest_sha256,
            key_label=key_label,
            include_public_key=include_public_key,
        )
        signature_path = pack_root / SIGNATURE_FILENAME
        signature_path.write_text(json.dumps(signature_doc, indent=2, ensure_ascii=False), encoding="utf-8")
        return AuditSignResult(
            source=source_path,
            mode="dir",
            signature_path=str(signature_path),
            key_id=str(signature_doc["signed"]["key_id"]),
            checksums_sha256=checksums_sha256,
            manifest_sha256=manifest_sha256,
        )

    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        prefix, checksums_name, manifest_name = _resolve_pack_root_zip(source_path)
        with zipfile.ZipFile(source_path, "r") as zf:
            checksums_sha256 = _sha256_bytes(zf.read(checksums_name))
            manifest_sha256 = _sha256_bytes(zf.read(manifest_name))
        signature_doc = _build_signature_document(
            private_key=private_key,
            public_key=public_key,
            checksums_sha256=checksums_sha256,
            manifest_sha256=manifest_sha256,
            key_label=key_label,
            include_public_key=include_public_key,
        )
        signature_name = f"{prefix}/{SIGNATURE_FILENAME}" if prefix else SIGNATURE_FILENAME
        _write_zip_member(
            source_path,
            signature_name,
            json.dumps(signature_doc, indent=2, ensure_ascii=False).encode("utf-8"),
        )
        return AuditSignResult(
            source=source_path,
            mode="zip",
            signature_path=f"{source_path}!{signature_name}",
            key_id=str(signature_doc["signed"]["key_id"]),
            checksums_sha256=checksums_sha256,
            manifest_sha256=manifest_sha256,
        )

    raise FileNotFoundError(f"pack path not found or unsupported: {source_path}")


def resolve_public_key_for_verification(
    *,
    public_key_path: str | Path | None,
    public_key_dir: str | Path | None,
    signature_doc: dict[str, Any],
    trust_embedded_public_key: bool = False,
) -> tuple[Ed25519PublicKey | None, str]:
    """Resolve verification key from CLI path or embedded key in signature.json."""
    if public_key_path is not None and str(public_key_path).strip() != "":
        path = Path(public_key_path).expanduser().resolve()
        return _load_public_key(path), str(path)

    if public_key_dir is not None and str(public_key_dir).strip() != "":
        directory = Path(public_key_dir).expanduser().resolve()
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"public key directory not found: {directory}")

        key_id = _signed_key_id(signature_doc).lower()
        if key_id == "":
            raise ValueError("signature payload does not include key_id")

        pem_paths = sorted(
            p for p in directory.rglob("*.pem")
            if p.is_file()
        )
        if not pem_paths:
            raise FileNotFoundError(f"no .pem files found in public key directory: {directory}")

        for pem_path in pem_paths:
            try:
                key = _load_public_key(pem_path)
            except Exception:
                continue
            if _key_id(key).lower() == key_id:
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
    if str(signature_doc.get("algorithm", "")).lower() != SIGNATURE_ALGORITHM:
        return False, f"unsupported algorithm: {signature_doc.get('algorithm')}"

    signed = signature_doc.get("signed")
    if not isinstance(signed, dict):
        return False, "missing signed payload"

    expected_checksums = str(signed.get("checksums_sha256", "")).strip().lower()
    expected_manifest = str(signed.get("manifest_sha256", "")).strip().lower()
    if expected_checksums != checksums_sha256.lower():
        return False, "signed checksums_sha256 does not match actual checksums file hash"
    if expected_manifest != manifest_sha256.lower():
        return False, "signed manifest_sha256 does not match actual manifest hash"

    sig_text = str(signature_doc.get("signature", "")).strip()
    if sig_text == "":
        return False, "missing signature"
    try:
        sig_bytes = base64.b64decode(sig_text, validate=True)
    except Exception as exc:
        return False, f"invalid base64 signature: {exc}"

    try:
        public_key.verify(sig_bytes, _canonical_json_bytes(signed))
    except InvalidSignature:
        return False, "invalid signature"
    except Exception as exc:
        return False, f"signature verification failed: {exc}"

    expected_key_id = str(signed.get("key_id", "")).strip()
    if expected_key_id == "":
        return False, "missing key_id in signed payload"
    if expected_key_id != _key_id(public_key):
        return False, "key_id does not match verification key"

    return True, ""
