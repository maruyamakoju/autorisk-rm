"""Internal helpers to locate audit pack roots in directory or zip forms."""

from __future__ import annotations

import zipfile
from pathlib import Path


def resolve_pack_root_dir(pack_dir: Path, *, checksums_filename: str) -> tuple[Path, Path]:
    candidates = sorted(pack_dir.rglob(checksums_filename), key=lambda p: (len(p.parts), str(p)))
    if not candidates:
        raise FileNotFoundError(f"missing {checksums_filename} under: {pack_dir}")
    checksums_path = candidates[0]
    pack_root = checksums_path.parent
    return pack_root, checksums_path


def resolve_pack_root_zip(zip_path: Path, *, checksums_filename: str) -> tuple[str, str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [n for n in zf.namelist() if n.endswith(checksums_filename) and not n.endswith("/")]
        if not members:
            raise FileNotFoundError(f"missing {checksums_filename} inside zip: {zip_path}")
        checksums_name = min(members, key=lambda n: (n.count("/"), len(n)))
        prefix = checksums_name[: -len(checksums_filename)].rstrip("/")
    return prefix, checksums_name
