"""Verify handoff folders end-to-end with one command."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from autorisk.audit.validate import validate_audit_pack
from autorisk.audit.verify import verify_audit_pack
from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)

HANDOFF_CHECKSUMS_FILENAME = "handoff_checksums.sha256.txt"
PACK_FINALIZE_RECORD_REL = "run_artifacts/finalize_record.json"
_LINE = re.compile(r"^([0-9a-fA-F]{64})\s{2}(.+)$")
_WINDOWS_DRIVE = re.compile(r"^[a-zA-Z]:")
_EXPECTED_VALIDATE_REPORT_PATH = "run_artifacts/audit_validate_report.json"
_MAX_BUNDLE_FILES = 1024
_MAX_BUNDLE_FILE_SIZE = 32 * 1024 * 1024
_MAX_BUNDLE_TOTAL_SIZE = 256 * 1024 * 1024


@dataclass
class HandoffVerifyIssue:
    kind: str  # parse_error | missing_file | mismatch | io_error | security_error | audit_verify_error | audit_validate_error | report_mismatch | finalize_record_error
    path: str
    detail: str = ""
    expected_sha256: str | None = None
    actual_sha256: str | None = None


@dataclass
class AuditHandoffVerifyResult:
    handoff_dir: Path
    checksums_path: Path
    listed_files: int
    verified_files: int
    pack_path: Path
    verifier_bundle_zip_path: Path
    finalize_record_path: Path
    validate_report_path: Path | None
    audit_verify_ok: bool
    audit_validate_ok: bool
    attestation_present: bool
    attestation_verified: bool | None
    attestation_key_id: str
    attestation_key_source: str
    bundled_validate_report_match: bool | None
    issues: list[HandoffVerifyIssue]

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "handoff_dir": str(self.handoff_dir),
            "checksums_path": str(self.checksums_path),
            "listed_files": int(self.listed_files),
            "verified_files": int(self.verified_files),
            "pack_path": str(self.pack_path),
            "verifier_bundle_zip_path": str(self.verifier_bundle_zip_path),
            "finalize_record_path": str(self.finalize_record_path),
            "validate_report_path": str(self.validate_report_path) if self.validate_report_path is not None else "",
            "audit_verify_ok": bool(self.audit_verify_ok),
            "audit_validate_ok": bool(self.audit_validate_ok),
            "attestation_present": bool(self.attestation_present),
            "attestation_verified": self.attestation_verified,
            "attestation_key_id": self.attestation_key_id,
            "attestation_key_source": self.attestation_key_source,
            "bundled_validate_report_match": self.bundled_validate_report_match,
            "ok": self.ok,
            "issues": [asdict(issue) for issue in self.issues],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_checksums(path: Path) -> tuple[list[tuple[str, str]], list[HandoffVerifyIssue]]:
    entries: list[tuple[str, str]] = []
    issues: list[HandoffVerifyIssue] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if line == "" or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            issues.append(
                HandoffVerifyIssue(
                    kind="parse_error",
                    path=str(path),
                    detail=f"line {lineno}: {line[:200]}",
                )
            )
            continue
        entries.append((match.group(1).lower(), match.group(2)))
    return entries, issues


def _load_revoked_key_ids(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.exists() or not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line == "" or line.startswith("#"):
            continue
        out.add(line.split()[0].lower())
    return out


def _safe_extract_zip(
    *,
    zip_path: Path,
    output_dir: Path,
    issues: list[HandoffVerifyIssue],
) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    ok = True

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            if len(infos) > _MAX_BUNDLE_FILES:
                issues.append(
                    HandoffVerifyIssue(
                        kind="security_error",
                        path=str(zip_path),
                        detail=f"too many files in verifier bundle: {len(infos)} > {_MAX_BUNDLE_FILES}",
                    )
                )
                return False

            total_size = 0
            for info in infos:
                name = info.filename.replace("\\", "/")
                if name.startswith("/") or _WINDOWS_DRIVE.match(name):
                    issues.append(
                        HandoffVerifyIssue(
                            kind="security_error",
                            path=name,
                            detail="absolute path not allowed in verifier_bundle.zip",
                        )
                    )
                    ok = False
                    continue

                rel_path = Path(name)
                if ".." in rel_path.parts:
                    issues.append(
                        HandoffVerifyIssue(
                            kind="security_error",
                            path=name,
                            detail="path traversal (..) not allowed in verifier_bundle.zip",
                        )
                    )
                    ok = False
                    continue

                if info.file_size > _MAX_BUNDLE_FILE_SIZE:
                    issues.append(
                        HandoffVerifyIssue(
                            kind="security_error",
                            path=name,
                            detail=f"file too large: {info.file_size} > {_MAX_BUNDLE_FILE_SIZE}",
                        )
                    )
                    ok = False
                    continue

                total_size += int(max(info.file_size, 0))
                if total_size > _MAX_BUNDLE_TOTAL_SIZE:
                    issues.append(
                        HandoffVerifyIssue(
                            kind="security_error",
                            path=str(zip_path),
                            detail=f"total extracted size exceeds limit: {total_size} > {_MAX_BUNDLE_TOTAL_SIZE}",
                        )
                    )
                    return False

                target = (output_root / rel_path).resolve()
                if output_root != target and output_root not in target.parents:
                    issues.append(
                        HandoffVerifyIssue(
                            kind="security_error",
                            path=name,
                            detail="resolved path escapes extraction root",
                        )
                    )
                    ok = False
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    except Exception as exc:
        issues.append(
            HandoffVerifyIssue(
                kind="io_error",
                path=str(zip_path),
                detail=str(exc)[:200],
            )
        )
        return False

    return ok


def _append_finalize_record_checks(
    *,
    record: dict[str, Any],
    verifier_bundle_zip_path: Path,
    checksums_path: Path,
    bundled_validate_path: Path | None,
    source_label: str,
    issues: list[HandoffVerifyIssue],
) -> None:
    path_label = source_label

    def _cmp(expected_value: str, field_name: str, actual: str, reference: str) -> None:
        expected = str(expected_value).strip().lower()
        if expected == "":
            issues.append(
                HandoffVerifyIssue(
                    kind="finalize_record_error",
                    path=path_label,
                    detail=f"{field_name} is missing",
                )
            )
            return
        if expected != actual.lower():
            issues.append(
                HandoffVerifyIssue(
                    kind="finalize_record_error",
                    path=path_label,
                    detail=f"{field_name} mismatch vs {reference}",
                    expected_sha256=expected,
                    actual_sha256=actual,
                )
            )

    expected_bundle = str(record.get("handoff_anchor_verifier_bundle_zip_sha256", "")).strip()
    if expected_bundle == "":
        expected_bundle = str(record.get("handoff_verifier_bundle_zip_sha256", "")).strip()
    expected_checksums = str(record.get("handoff_anchor_checksums_sha256", "")).strip()
    if expected_checksums == "":
        expected_checksums = str(record.get("handoff_checksums_sha256", "")).strip()

    _cmp(expected_bundle, "handoff_anchor_verifier_bundle_zip_sha256", _sha256_file(verifier_bundle_zip_path), "verifier_bundle.zip")
    _cmp(expected_checksums, "handoff_anchor_checksums_sha256", _sha256_file(checksums_path), HANDOFF_CHECKSUMS_FILENAME)

    validate_report_path = str(record.get("validate_report_path", "")).strip().replace("\\", "/")
    if validate_report_path != "" and validate_report_path != _EXPECTED_VALIDATE_REPORT_PATH:
        issues.append(
            HandoffVerifyIssue(
                kind="finalize_record_error",
                path=path_label,
                detail=f"validate_report_path must be {_EXPECTED_VALIDATE_REPORT_PATH}",
            )
        )

    if bundled_validate_path is not None and bundled_validate_path.exists() and bundled_validate_path.is_file():
        _cmp(
            str(record.get("validate_report_sha256", "")).strip(),
            "validate_report_sha256",
            _sha256_file(bundled_validate_path),
            "audit_validate_report.json",
        )

def _load_pack_finalize_record(
    *,
    pack_path: Path,
    issues: list[HandoffVerifyIssue],
) -> dict[str, Any] | None:
    try:
        with zipfile.ZipFile(pack_path, "r") as zf:
            checksum_members = [
                n for n in zf.namelist()
                if n.endswith("checksums.sha256.txt") and not n.endswith("/")
            ]
            if not checksum_members:
                issues.append(
                    HandoffVerifyIssue(
                        kind="missing_file",
                        path="PACK.zip!checksums.sha256.txt",
                        detail="pack checksums not found while reading attested finalize record",
                    )
                )
                return None
            checksums_name = min(checksum_members, key=lambda n: (n.count("/"), len(n)))
            prefix = checksums_name[: -len("checksums.sha256.txt")].rstrip("/")
            root_prefix = f"{prefix}/" if prefix else ""
            finalize_name = f"{root_prefix}{PACK_FINALIZE_RECORD_REL}" if root_prefix else PACK_FINALIZE_RECORD_REL
            if finalize_name not in set(zf.namelist()):
                issues.append(
                    HandoffVerifyIssue(
                        kind="missing_file",
                        path=f"PACK.zip!{finalize_name}",
                        detail="attested pack finalize_record is missing",
                    )
                )
                return None
            loaded = json.loads(zf.read(finalize_name).decode("utf-8", errors="replace"))
            if not isinstance(loaded, dict):
                issues.append(
                    HandoffVerifyIssue(
                        kind="parse_error",
                        path=f"PACK.zip!{finalize_name}",
                        detail="pack finalize_record must be a JSON object",
                    )
                )
                return None
            return loaded
    except Exception as exc:
        issues.append(
            HandoffVerifyIssue(
                kind="io_error",
                path="PACK.zip",
                detail=f"failed to read attested finalize_record: {str(exc)[:200]}",
            )
        )
        return None


def _compare_handoff_finalize_copy(
    *,
    handoff_finalize_record: dict[str, Any] | None,
    pack_finalize_record: dict[str, Any] | None,
    issues: list[HandoffVerifyIssue],
) -> None:
    if handoff_finalize_record is None or pack_finalize_record is None:
        return
    for field in [
        "pack_fingerprint",
        "signature_key_id",
        "validate_ok",
        "validate_issues_count",
        "validate_report_sha256",
    ]:
        left = handoff_finalize_record.get(field)
        right = pack_finalize_record.get(field)
        if left != right:
            issues.append(
                HandoffVerifyIssue(
                    kind="finalize_record_error",
                    path="finalize_record.json",
                    detail=f"handoff finalize copy diverges from attested pack finalize at field={field}",
                )
            )


def verify_audit_handoff(
    handoff_dir: str | Path,
    *,
    strict: bool = True,
    require_signature: bool = True,
    require_public_key: bool = True,
    require_attestation: bool = True,
    validate_profile: str = "audit-grade",
    compare_bundled_validate_report: bool = True,
) -> AuditHandoffVerifyResult:
    """Verify handoff checksums, then verify and validate bundled PACK.zip."""
    handoff_dir_path = Path(handoff_dir).expanduser().resolve()
    if not handoff_dir_path.exists() or not handoff_dir_path.is_dir():
        raise FileNotFoundError(f"handoff directory not found: {handoff_dir_path}")

    checksums_path = handoff_dir_path / HANDOFF_CHECKSUMS_FILENAME
    if not checksums_path.exists() or not checksums_path.is_file():
        raise FileNotFoundError(f"missing {HANDOFF_CHECKSUMS_FILENAME}: {checksums_path}")

    issues: list[HandoffVerifyIssue] = []
    entries, parse_issues = _parse_checksums(checksums_path)
    issues.extend(parse_issues)

    forbidden_handoff_checksum_rows = {"PACK.zip", "finalize_record.json"}
    for _, rel in entries:
        if rel in forbidden_handoff_checksum_rows:
            issues.append(
                HandoffVerifyIssue(
                    kind="security_error",
                    path=rel,
                    detail="handoff_checksums must not include this file (contract requires PACK/finalize exclusion)",
                )
            )

    verified_files = 0
    for expected_sha, rel in entries:
        target = handoff_dir_path / rel
        if not target.exists() or not target.is_file():
            issues.append(
                HandoffVerifyIssue(
                    kind="missing_file",
                    path=rel,
                    expected_sha256=expected_sha,
                )
            )
            continue
        try:
            actual_sha = _sha256_file(target)
        except Exception as exc:
            issues.append(HandoffVerifyIssue(kind="io_error", path=rel, detail=str(exc)[:200]))
            continue
        verified_files += 1
        if actual_sha.lower() != expected_sha.lower():
            issues.append(
                HandoffVerifyIssue(
                    kind="mismatch",
                    path=rel,
                    expected_sha256=expected_sha,
                    actual_sha256=actual_sha,
                )
            )

    pack_path = handoff_dir_path / "PACK.zip"
    if not pack_path.exists() or not pack_path.is_file():
        issues.append(HandoffVerifyIssue(kind="missing_file", path="PACK.zip", detail="required file is missing"))

    verifier_bundle_zip_path = handoff_dir_path / "verifier_bundle.zip"
    if not verifier_bundle_zip_path.exists() or not verifier_bundle_zip_path.is_file():
        issues.append(HandoffVerifyIssue(kind="missing_file", path="verifier_bundle.zip", detail="required file is missing"))

    finalize_record_path = handoff_dir_path / "finalize_record.json"
    if not finalize_record_path.exists() or not finalize_record_path.is_file():
        issues.append(HandoffVerifyIssue(kind="missing_file", path="finalize_record.json", detail="required file is missing"))

    bundled_validate_path = handoff_dir_path / "audit_validate_report.json"
    if compare_bundled_validate_report and (not bundled_validate_path.exists() or not bundled_validate_path.is_file()):
        issues.append(
            HandoffVerifyIssue(
                kind="missing_file",
                path="audit_validate_report.json",
                detail="required for bundled validate comparison",
            )
        )

    audit_verify_ok = False
    audit_validate_ok = False
    attestation_present = False
    attestation_verified: bool | None = None
    attestation_key_id = ""
    attestation_key_source = ""
    bundled_validate_report_match: bool | None = None
    finalize_record_obj: dict[str, Any] | None = None
    pack_finalize_record_obj: dict[str, Any] | None = None

    if finalize_record_path.exists() and finalize_record_path.is_file():
        try:
            loaded = json.loads(finalize_record_path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(loaded, dict):
                finalize_record_obj = loaded
            else:
                issues.append(
                    HandoffVerifyIssue(
                        kind="parse_error",
                        path="finalize_record.json",
                        detail="finalize_record.json must be an object",
                    )
                )
        except Exception as exc:
            issues.append(
                HandoffVerifyIssue(
                    kind="parse_error",
                    path="finalize_record.json",
                    detail=str(exc)[:200],
                )
            )

    if pack_path.exists() and verifier_bundle_zip_path.exists():
        with tempfile.TemporaryDirectory(prefix="autorisk-handoff-verify-") as tmp_dir:
            bundle_root = Path(tmp_dir) / "verifier_bundle"
            safe_extract_ok = _safe_extract_zip(
                zip_path=verifier_bundle_zip_path,
                output_dir=bundle_root,
                issues=issues,
            )
            if not safe_extract_ok:
                return AuditHandoffVerifyResult(
                    handoff_dir=handoff_dir_path,
                    checksums_path=checksums_path,
                    listed_files=len(entries),
                    verified_files=verified_files,
                    pack_path=pack_path,
                    verifier_bundle_zip_path=verifier_bundle_zip_path,
                    finalize_record_path=finalize_record_path,
                    validate_report_path=bundled_validate_path if bundled_validate_path.exists() else None,
                    audit_verify_ok=audit_verify_ok,
                    audit_validate_ok=audit_validate_ok,
                    attestation_present=attestation_present,
                    attestation_verified=attestation_verified,
                    attestation_key_id=attestation_key_id,
                    attestation_key_source=attestation_key_source,
                    bundled_validate_report_match=bundled_validate_report_match,
                    issues=issues,
                )
            trusted_keys_dir = bundle_root / "keys" / "trusted"
            revocation_file = bundle_root / "revoked_key_ids.txt"
            revoked = _load_revoked_key_ids(revocation_file)

            if not trusted_keys_dir.exists() or not trusted_keys_dir.is_dir():
                issues.append(HandoffVerifyIssue(kind="missing_file", path=str(trusted_keys_dir), detail="missing trusted key directory"))

            verify_res = verify_audit_pack(
                pack_path,
                strict=strict,
                public_key_dir=trusted_keys_dir if trusted_keys_dir.exists() else None,
                require_signature=require_signature,
                require_public_key=require_public_key,
                require_attestation=require_attestation,
                revoked_key_ids=revoked,
            )
            audit_verify_ok = verify_res.ok
            attestation_present = bool(verify_res.attestation_present)
            attestation_verified = verify_res.attestation_verified
            attestation_key_id = str(verify_res.attestation_key_id or "")
            attestation_key_source = str(verify_res.attestation_key_source or "")
            for issue in verify_res.issues:
                issues.append(
                    HandoffVerifyIssue(
                        kind="audit_verify_error",
                        path=issue.path,
                        detail=f"{issue.kind}: {issue.detail}".strip(),
                        expected_sha256=issue.expected_sha256,
                        actual_sha256=issue.actual_sha256,
                    )
                )
            pack_finalize_record_obj = _load_pack_finalize_record(
                pack_path=pack_path,
                issues=issues,
            )
            _compare_handoff_finalize_copy(
                handoff_finalize_record=finalize_record_obj,
                pack_finalize_record=pack_finalize_record_obj,
                issues=issues,
            )

            validate_res = validate_audit_pack(
                pack_path,
                semantic_checks=True,
                profile=validate_profile,
            )
            audit_validate_ok = validate_res.ok
            for issue in validate_res.issues:
                suffix = f" line={issue.line}" if issue.line is not None else ""
                issues.append(
                    HandoffVerifyIssue(
                        kind="audit_validate_error",
                        path=issue.path,
                        detail=f"{issue.kind}: {issue.detail}{suffix}".strip(),
                    )
                )

            if compare_bundled_validate_report and bundled_validate_path.exists() and bundled_validate_path.is_file():
                try:
                    bundled_obj = json.loads(bundled_validate_path.read_text(encoding="utf-8"))
                    bundled_ok = bool(bundled_obj.get("ok", False))
                    bundled_issues = bundled_obj.get("issues", [])
                    bundled_issue_count = len(bundled_issues) if isinstance(bundled_issues, list) else None
                    bundled_validate_report_match = (
                        bundled_ok == validate_res.ok
                        and bundled_issue_count == len(validate_res.issues)
                    )
                    if not bundled_validate_report_match:
                        issues.append(
                            HandoffVerifyIssue(
                                kind="report_mismatch",
                                path="audit_validate_report.json",
                                detail=(
                                    "bundled report mismatch: "
                                    f"bundled(ok={bundled_ok}, issues={bundled_issue_count}) "
                                    f"vs recomputed(ok={validate_res.ok}, issues={len(validate_res.issues)})"
                                ),
                            )
                        )
                except Exception as exc:
                    issues.append(
                        HandoffVerifyIssue(
                            kind="io_error",
                            path="audit_validate_report.json",
                            detail=str(exc)[:200],
                        )
                    )

    if (
        pack_finalize_record_obj is not None
        and attestation_verified is True
        and verifier_bundle_zip_path.exists()
        and checksums_path.exists()
    ):
        _append_finalize_record_checks(
            record=pack_finalize_record_obj,
            verifier_bundle_zip_path=verifier_bundle_zip_path,
            checksums_path=checksums_path,
            bundled_validate_path=bundled_validate_path if bundled_validate_path.exists() else None,
            source_label=f"PACK.zip!{PACK_FINALIZE_RECORD_REL}",
            issues=issues,
        )
    elif pack_finalize_record_obj is not None and attestation_verified is not True:
        issues.append(
            HandoffVerifyIssue(
                kind="finalize_record_error",
                path=f"PACK.zip!{PACK_FINALIZE_RECORD_REL}",
                detail="cannot trust handoff anchors because PACK attestation is not verified",
            )
        )

    result = AuditHandoffVerifyResult(
        handoff_dir=handoff_dir_path,
        checksums_path=checksums_path,
        listed_files=len(entries),
        verified_files=verified_files,
        pack_path=pack_path,
        verifier_bundle_zip_path=verifier_bundle_zip_path,
        finalize_record_path=finalize_record_path,
        validate_report_path=bundled_validate_path if bundled_validate_path.exists() else None,
        audit_verify_ok=audit_verify_ok,
        audit_validate_ok=audit_validate_ok,
        attestation_present=attestation_present,
        attestation_verified=attestation_verified,
        attestation_key_id=attestation_key_id,
        attestation_key_source=attestation_key_source,
        bundled_validate_report_match=bundled_validate_report_match,
        issues=issues,
    )
    log.info(
        "Handoff verify: dir=%s listed=%d verified=%d issues=%d",
        result.handoff_dir,
        result.listed_files,
        result.verified_files,
        len(result.issues),
    )
    return result
