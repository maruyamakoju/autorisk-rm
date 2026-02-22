"""Internal cryptographic and hashing helpers for audit modules."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

_SHA256_CHUNK_SIZE = 1024 * 1024


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_stream(fp: BinaryIO) -> str:
    h = hashlib.sha256()
    for chunk in iter(lambda: fp.read(_SHA256_CHUNK_SIZE), b""):
        h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return sha256_stream(f)


def load_private_key(path: Path, *, password: str | bytes | None = None) -> Ed25519PrivateKey:
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


def load_public_key(path: Path) -> Ed25519PublicKey:
    key_bytes = path.read_bytes()
    key = serialization.load_pem_public_key(key_bytes)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError(f"public key is not Ed25519: {path}")
    return key


def public_key_pem(public_key: Ed25519PublicKey) -> str:
    return (
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
        .strip()
    )


def public_key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()[:16]


def assert_matching_keypair(
    *,
    private_key: Ed25519PrivateKey,
    public_key: Ed25519PublicKey,
    context: str,
) -> None:
    derived = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    supplied = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    if derived != supplied:
        raise ValueError(
            f"{context}: provided public key does not match the private key"
        )
