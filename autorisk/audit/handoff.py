"""Build single-file handoff packages for audit submission and receipt."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)


@dataclass
class AuditHandoffResult:
    output_dir: Path
    pack_zip_path: Path
    verifier_bundle_zip_path: Path
    finalize_record_path: Path
    validate_report_path: Path | None
    handoff_guide_path: Path
    checksums_path: Path


def _utc_now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _zip_dir(source_dir: Path, output_zip: Path) -> None:
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(source_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(source_dir).as_posix())


def _write_checksums(output_dir: Path) -> Path:
    rows: list[tuple[str, str]] = []
    for p in sorted(output_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(output_dir).as_posix()
        if rel in {"handoff_checksums.sha256.txt", "finalize_record.json", "PACK.zip"}:
            continue
        rows.append((_sha256_file(p), rel))
    checksums_path = output_dir / "handoff_checksums.sha256.txt"
    checksums_path.write_text(
        "\n".join(f"{sha}  {rel}" for sha, rel in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return checksums_path


def _latest_audit_zip(run_dir: Path) -> Path:
    zips = sorted(run_dir.glob("audit_pack_*.zip"), key=lambda p: p.name)
    if not zips:
        raise FileNotFoundError(f"no audit_pack_*.zip found in run dir: {run_dir}")
    return zips[-1]


def build_audit_handoff(
    *,
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    pack_zip: str | Path | None = None,
    verifier_bundle_dir: str | Path | None = None,
    finalize_record: str | Path | None = None,
    validate_report: str | Path | None = None,
) -> AuditHandoffResult:
    """Build a single handoff directory with pack, verifier bundle, and metadata."""
    run_dir_path = Path(run_dir).expanduser().resolve()
    if not run_dir_path.exists():
        raise FileNotFoundError(f"run directory not found: {run_dir_path}")

    pack_zip_path = (
        Path(pack_zip).expanduser().resolve()
        if pack_zip is not None and str(pack_zip).strip() != ""
        else _latest_audit_zip(run_dir_path)
    )
    if not pack_zip_path.exists() or not pack_zip_path.is_file():
        raise FileNotFoundError(f"pack zip not found: {pack_zip_path}")

    verifier_dir_path = (
        Path(verifier_bundle_dir).expanduser().resolve()
        if verifier_bundle_dir is not None and str(verifier_bundle_dir).strip() != ""
        else (run_dir_path / "verifier_bundle")
    )
    if not verifier_dir_path.exists() or not verifier_dir_path.is_dir():
        raise FileNotFoundError(f"verifier bundle directory not found: {verifier_dir_path}")

    finalize_record_path_src = (
        Path(finalize_record).expanduser().resolve()
        if finalize_record is not None and str(finalize_record).strip() != ""
        else (run_dir_path / "finalize_record.json")
    )
    if not finalize_record_path_src.exists() or not finalize_record_path_src.is_file():
        raise FileNotFoundError(f"finalize record not found: {finalize_record_path_src}")

    if output_dir is None:
        handoff_dir = run_dir_path / f"handoff_{_utc_now_slug()}"
    else:
        handoff_dir = Path(output_dir).expanduser().resolve()
    handoff_dir.mkdir(parents=True, exist_ok=True)

    pack_dst = handoff_dir / "PACK.zip"
    shutil.copy2(pack_zip_path, pack_dst)

    verifier_zip_dst = handoff_dir / "verifier_bundle.zip"
    _zip_dir(verifier_dir_path, verifier_zip_dst)

    finalize_dst = handoff_dir / "finalize_record.json"
    shutil.copy2(finalize_record_path_src, finalize_dst)

    validate_report_dst: Path | None = None
    validate_report_path_src = (
        Path(validate_report).expanduser().resolve()
        if validate_report is not None and str(validate_report).strip() != ""
        else (run_dir_path / "audit_validate_report.json")
    )
    if validate_report_path_src.exists() and validate_report_path_src.is_file():
        validate_report_dst = handoff_dir / "audit_validate_report.json"
        shutil.copy2(validate_report_path_src, validate_report_dst)

    handoff_md = handoff_dir / "HANDOFF.md"
    handoff_md.write_text(
        "\n".join(
            [
                "# Audit Handoff",
                "",
                "Before verification, obtain `pack_fingerprint` from the submission ticket/DB record.",
                "It must be the SHA256 of PACK checksums.sha256.txt captured at submission time.",
                "",
                "1. One-command verification (recommended):",
                "```bash",
                "python -m autorisk.cli audit-handoff-verify -d . --profile audit-grade \\",
                "  --expect-pack-fingerprint <TICKET_FP> --enforce",
                "```",
                "",
                "2. Manual verification steps (equivalent):",
                "",
                "Note: handoff_checksums.sha256.txt intentionally excludes PACK.zip and finalize_record.json.",
                "PACK.zip authenticity/integrity is enforced in step 2.2 via audit-verify.",
                "",
                "2.1 Unzip verifier bundle:",
                "```bash",
                "unzip verifier_bundle.zip -d verifier_bundle",
                "```",
                "",
                "2.2 Verify audit pack:",
                "```bash",
                "python -m autorisk.cli audit-verify -p PACK.zip --profile audit-grade \\",
                "  --public-key-dir verifier_bundle/keys/trusted \\",
                "  --revocation-file verifier_bundle/revoked_key_ids.txt \\",
                "  --expect-pack-fingerprint <TICKET_FP>",
                "```",
                "",
                "2.3 Validate audit contract:",
                "```bash",
                "python -m autorisk.cli audit-validate -p PACK.zip --profile audit-grade --enforce",
                "```",
                "",
                "3. Optional: inspect bundled validation report:",
                "```bash",
                "cat audit_validate_report.json",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    checksums_path = _write_checksums(handoff_dir)
    log.info("Audit handoff created: %s", handoff_dir)
    return AuditHandoffResult(
        output_dir=handoff_dir,
        pack_zip_path=pack_dst,
        verifier_bundle_zip_path=verifier_zip_dst,
        finalize_record_path=finalize_dst,
        validate_report_path=validate_report_dst,
        handoff_guide_path=handoff_md,
        checksums_path=checksums_path,
    )
