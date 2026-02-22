"""Internal zip mutation helpers for audit modules."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path


def rewrite_zip_member(
    *,
    zip_path: Path,
    member_name: str,
    payload: bytes,
    temp_prefix: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix=temp_prefix) as tmp_dir:
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
