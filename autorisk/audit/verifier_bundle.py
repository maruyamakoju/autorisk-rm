"""Build verifier bundle artifacts for third-party audit verification."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from autorisk.audit._crypto import sha256_file

@dataclass
class VerifierBundleResult:
    output_dir: Path
    trusted_keys_dir: Path
    verify_md_path: Path
    revocation_path: Path
    key_files: list[Path]
    revoked_key_ids: list[str]
def _normalized_revoked_ids(
    *,
    revoked_key_ids: set[str] | None,
    revocation_file: str | Path | None,
) -> list[str]:
    merged = {str(v).strip().lower() for v in (revoked_key_ids or set()) if str(v).strip() != ""}
    if revocation_file is not None and str(revocation_file).strip() != "":
        path = Path(revocation_file).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"revocation file not found: {path}")
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if line == "" or line.startswith("#"):
                continue
            merged.add(line.split()[0].lower())
    return sorted(merged)


def _copy_key(src: Path, dst_dir: Path, *, used_names: set[str]) -> Path:
    base_name = src.name
    candidate = base_name
    idx = 1
    while candidate.lower() in used_names:
        stem = src.stem
        suffix = src.suffix
        candidate = f"{stem}_{idx}{suffix}"
        idx += 1
    used_names.add(candidate.lower())
    dst = dst_dir / candidate
    dst.write_bytes(src.read_bytes())
    return dst


def _autorisk_version() -> str:
    try:
        return str(importlib.metadata.version("autorisk-rm"))
    except Exception:
        return "unknown"


def build_verifier_bundle(
    *,
    output_dir: str | Path,
    public_key: str | Path | None = None,
    public_key_dir: str | Path | None = None,
    revoked_key_ids: set[str] | None = None,
    revocation_file: str | Path | None = None,
    verify_pack_reference: str = "PACK_OR_ZIP",
) -> VerifierBundleResult:
    """Build a portable verifier bundle: trusted keys + revocation list + VERIFY.md."""
    out_dir = Path(output_dir).expanduser().resolve()
    trusted_dir = out_dir / "keys" / "trusted"
    trusted_dir.mkdir(parents=True, exist_ok=True)

    key_sources: list[Path] = []
    if public_key is not None and str(public_key).strip() != "":
        key_path = Path(public_key).expanduser().resolve()
        if not key_path.exists() or not key_path.is_file():
            raise FileNotFoundError(f"public key not found: {key_path}")
        key_sources.append(key_path)
    if public_key_dir is not None and str(public_key_dir).strip() != "":
        key_dir = Path(public_key_dir).expanduser().resolve()
        if not key_dir.exists() or not key_dir.is_dir():
            raise FileNotFoundError(f"public key directory not found: {key_dir}")
        key_sources.extend(sorted(p for p in key_dir.rglob("*.pem") if p.is_file()))

    if not key_sources:
        raise ValueError("at least one trusted key is required to build verifier bundle")

    used_names: set[str] = set()
    key_files: list[Path] = []
    for src in key_sources:
        key_files.append(_copy_key(src, trusted_dir, used_names=used_names))

    revoked = _normalized_revoked_ids(
        revoked_key_ids=revoked_key_ids,
        revocation_file=revocation_file,
    )
    revocation_path = out_dir / "revoked_key_ids.txt"
    revocation_path.write_text(
        "\n".join(revoked) + ("\n" if revoked else ""),
        encoding="utf-8",
    )

    verify_md_path = out_dir / "VERIFY.md"
    verify_md_path.write_text(
        "\n".join(
            [
                "# Verify Audit Pack",
                "",
                "Run this command:",
                "",
                "```bash",
                f"python -m autorisk.cli audit-verify -p {verify_pack_reference} --strict \\",
                "  --public-key-dir keys/trusted --revocation-file revoked_key_ids.txt \\",
                "  --require-signature --require-public-key",
                "```",
                "",
                "Notes:",
                f"- trusted key files: {len(key_files)}",
                f"- revocation file sha256: {sha256_file(revocation_path)}",
                f"- autorisk version: {_autorisk_version()}",
                f"- python version: {sys.version.splitlines()[0]}",
                f"- platform: {platform.platform()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return VerifierBundleResult(
        output_dir=out_dir,
        trusted_keys_dir=trusted_dir,
        verify_md_path=verify_md_path,
        revocation_path=revocation_path,
        key_files=key_files,
        revoked_key_ids=revoked,
    )
