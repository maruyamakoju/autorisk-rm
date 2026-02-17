"""Verify audit packs by re-computing SHA256 checksums.

Supports both directories and zip bundles created by `autorisk audit-pack`.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from autorisk.audit.attestation import (
    ATTESTATION_FILENAME,
    FINALIZE_RECORD_REL,
    VALIDATE_REPORT_REL,
    verify_attestation_document,
)
from autorisk.audit.sign import (
    MANIFEST_FILENAME,
    SIGNATURE_FILENAME,
    resolve_public_key_for_verification,
    verify_signature_document,
)
from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)

CHECKSUMS_FILENAME = "checksums.sha256.txt"
_OPTIONAL_UNCHECKSUMED_FILES = {
    ATTESTATION_FILENAME,
    "run_artifacts/finalize_record.json",
    "run_artifacts/audit_validate_report.json",
}


@dataclass
class VerifyIssue:
    kind: str  # mismatch | missing_file | unexpected_file | parse_error | io_error | signature_error | attestation_error
    path: str
    expected_sha256: str | None = None
    actual_sha256: str | None = None
    detail: str = ""


@dataclass
class AuditVerifyResult:
    source: Path
    mode: str  # "dir" | "zip"
    pack_root: str  # directory path or zip prefix
    checksums_path: str
    checksums_sha256: str
    expected_files: int
    verified_files: int
    issues: list[VerifyIssue]
    signature_present: bool = False
    signature_path: str = ""
    signature_verified: bool | None = None
    signature_key_id: str = ""
    signature_key_source: str = ""
    attestation_present: bool = False
    attestation_path: str = ""
    attestation_verified: bool | None = None
    attestation_key_id: str = ""
    attestation_key_source: str = ""
    unchecked_files: list[str] | None = None

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "mode": self.mode,
            "pack_root": self.pack_root,
            "checksums_path": self.checksums_path,
            "checksums_sha256": self.checksums_sha256,
            "expected_files": int(self.expected_files),
            "verified_files": int(self.verified_files),
            "ok": self.ok,
            "signature_present": bool(self.signature_present),
            "signature_path": self.signature_path,
            "signature_verified": self.signature_verified,
            "signature_key_id": self.signature_key_id,
            "signature_key_source": self.signature_key_source,
            "attestation_present": bool(self.attestation_present),
            "attestation_path": self.attestation_path,
            "attestation_verified": self.attestation_verified,
            "attestation_key_id": self.attestation_key_id,
            "attestation_key_source": self.attestation_key_source,
            "unchecked_files": list(self.unchecked_files or []),
            "issues": [asdict(i) for i in self.issues],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_LINE = re.compile(r"^([0-9a-fA-F]{64})\s{2}(.+)$")


def _sha256_stream(fp) -> str:
    h = hashlib.sha256()
    for chunk in iter(lambda: fp.read(1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return _sha256_stream(f)


def _parse_checksums_text(
    text: str,
    *,
    display_path: str,
) -> tuple[list[tuple[str, str]], list[VerifyIssue]]:
    entries: list[tuple[str, str]] = []
    issues: list[VerifyIssue] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\n")
        if line.strip() == "" or line.lstrip().startswith("#"):
            continue

        m = _LINE.match(line.strip())
        if m:
            entries.append((m.group(1).lower(), m.group(2)))
            continue

        parts = line.strip().split()
        if len(parts) >= 2 and _HEX64.match(parts[0]):
            sha = parts[0].lower()
            rel = " ".join(parts[1:])
            entries.append((sha, rel))
            continue

        issues.append(
            VerifyIssue(
                kind="parse_error",
                path=display_path,
                detail=f"line {lineno}: {line[:200]}",
            )
        )
    return entries, issues


def _verify_signature(
    *,
    signature_present: bool,
    signature_doc: dict[str, Any] | None,
    signature_path: str,
    checksums_sha256: str,
    manifest_sha256: str,
    public_key: str | Path | None,
    public_key_dir: str | Path | None,
    require_signature: bool,
    require_public_key: bool,
    trust_embedded_public_key: bool,
    revoked_key_ids: set[str],
    issues: list[VerifyIssue],
) -> tuple[bool, bool | None, str, str]:
    signature_verified: bool | None = None
    key_id = ""
    key_source = ""

    has_explicit_public_key = (
        (public_key is not None and str(public_key).strip() != "")
        or (public_key_dir is not None and str(public_key_dir).strip() != "")
    )
    if require_public_key and not has_explicit_public_key:
        issues.append(
            VerifyIssue(
                kind="signature_error",
                path=signature_path,
                detail="--require-public-key set but no --public-key/--public-key-dir was provided",
            )
        )

    if not signature_present:
        if require_signature or has_explicit_public_key:
            issues.append(
                VerifyIssue(
                    kind="signature_error",
                    path=signature_path,
                    detail="signature.json not found",
                )
            )
            signature_verified = False
        return signature_present, signature_verified, key_id, key_source

    if signature_doc is None:
        issues.append(
            VerifyIssue(
                kind="signature_error",
                path=signature_path,
                detail="signature.json is unreadable",
            )
        )
        return signature_present, False, key_id, key_source

    signed = signature_doc.get("signed", {})
    if isinstance(signed, dict):
        key_id = str(signed.get("key_id", ""))
    normalized_key_id = key_id.strip().lower()
    if normalized_key_id != "" and normalized_key_id in revoked_key_ids:
        issues.append(
            VerifyIssue(
                kind="signature_error",
                path=signature_path,
                detail=f"signature key_id is revoked: {normalized_key_id}",
            )
        )
        return signature_present, False, key_id, key_source

    try:
        resolved_public_key, key_source = resolve_public_key_for_verification(
            public_key_path=public_key,
            public_key_dir=public_key_dir,
            signature_doc=signature_doc,
            trust_embedded_public_key=trust_embedded_public_key,
        )
    except Exception as exc:
        issues.append(
            VerifyIssue(
                kind="signature_error",
                path=signature_path,
                detail=str(exc)[:200],
            )
        )
        return signature_present, False, key_id, key_source

    if resolved_public_key is None:
        if require_public_key:
            return signature_present, False, key_id, key_source
        return signature_present, None, key_id, key_source

    ok, detail = verify_signature_document(
        signature_doc,
        checksums_sha256=checksums_sha256,
        manifest_sha256=manifest_sha256,
        public_key=resolved_public_key,
    )
    if not ok:
        issues.append(
            VerifyIssue(
                kind="signature_error",
                path=signature_path,
                detail=detail,
            )
        )
        return signature_present, False, key_id, key_source
    return signature_present, True, key_id, key_source


def _verify_attestation(
    *,
    attestation_present: bool,
    attestation_doc: dict[str, Any] | None,
    attestation_path: str,
    checksums_sha256: str,
    finalize_record_sha256: str,
    audit_validate_report_sha256: str,
    public_key: str | Path | None,
    public_key_dir: str | Path | None,
    require_attestation: bool,
    trust_embedded_public_key: bool,
    revoked_key_ids: set[str],
    issues: list[VerifyIssue],
) -> tuple[bool, bool | None, str, str]:
    attestation_verified: bool | None = None
    key_id = ""
    key_source = ""

    if not attestation_present:
        if require_attestation:
            issues.append(
                VerifyIssue(
                    kind="attestation_error",
                    path=attestation_path,
                    detail="attestation.json not found",
                )
            )
            attestation_verified = False
        return attestation_present, attestation_verified, key_id, key_source

    if attestation_doc is None:
        issues.append(
            VerifyIssue(
                kind="attestation_error",
                path=attestation_path,
                detail="attestation.json is unreadable",
            )
        )
        return attestation_present, False, key_id, key_source

    if finalize_record_sha256.strip() == "":
        issues.append(
            VerifyIssue(
                kind="attestation_error",
                path=attestation_path,
                detail=f"missing {FINALIZE_RECORD_REL} required for attestation verification",
            )
        )
        return attestation_present, False, key_id, key_source

    if audit_validate_report_sha256.strip() == "":
        issues.append(
            VerifyIssue(
                kind="attestation_error",
                path=attestation_path,
                detail=f"missing {VALIDATE_REPORT_REL} required for attestation verification",
            )
        )
        return attestation_present, False, key_id, key_source

    signed = attestation_doc.get("signed", {})
    if isinstance(signed, dict):
        key_id = str(signed.get("key_id", ""))
    normalized_key_id = key_id.strip().lower()
    if normalized_key_id != "" and normalized_key_id in revoked_key_ids:
        issues.append(
            VerifyIssue(
                kind="attestation_error",
                path=attestation_path,
                detail=f"attestation key_id is revoked: {normalized_key_id}",
            )
        )
        return attestation_present, False, key_id, key_source

    try:
        resolved_public_key, key_source = resolve_public_key_for_verification(
            public_key_path=public_key,
            public_key_dir=public_key_dir,
            signature_doc=attestation_doc,
            trust_embedded_public_key=trust_embedded_public_key,
        )
    except Exception as exc:
        issues.append(
            VerifyIssue(
                kind="attestation_error",
                path=attestation_path,
                detail=str(exc)[:200],
            )
        )
        return attestation_present, False, key_id, key_source

    if resolved_public_key is None:
        if require_attestation:
            issues.append(
                VerifyIssue(
                    kind="attestation_error",
                    path=attestation_path,
                    detail="attestation verification key is unavailable",
                )
            )
            return attestation_present, False, key_id, key_source
        return attestation_present, None, key_id, key_source

    ok, detail = verify_attestation_document(
        attestation_doc,
        pack_fingerprint=checksums_sha256,
        finalize_record_sha256=finalize_record_sha256,
        audit_validate_report_sha256=audit_validate_report_sha256,
        public_key=resolved_public_key,
    )
    if not ok:
        issues.append(
            VerifyIssue(
                kind="attestation_error",
                path=attestation_path,
                detail=detail,
            )
        )
        return attestation_present, False, key_id, key_source
    return attestation_present, True, key_id, key_source


def _verify_dir(
    pack_dir: Path,
    *,
    strict: bool = True,
    public_key: str | Path | None = None,
    public_key_dir: str | Path | None = None,
    require_signature: bool = False,
    require_public_key: bool = False,
    require_attestation: bool = False,
    trust_embedded_public_key: bool = False,
    revoked_key_ids: set[str] | None = None,
) -> AuditVerifyResult:
    pack_dir = pack_dir.resolve()
    candidates = sorted(pack_dir.rglob(CHECKSUMS_FILENAME), key=lambda p: (len(p.parts), str(p)))
    if not candidates:
        raise FileNotFoundError(f"missing {CHECKSUMS_FILENAME} under: {pack_dir}")

    checksums_path = candidates[0]
    pack_root = checksums_path.parent

    checksums_text = checksums_path.read_text(encoding="utf-8", errors="replace")
    checksums_sha256 = sha256_file(checksums_path)

    entries, parse_issues = _parse_checksums_text(checksums_text, display_path=str(checksums_path))
    issues: list[VerifyIssue] = list(parse_issues)
    unchecked_files: set[str] = set()

    expected_by_path: dict[str, str] = {}
    for sha, rel in entries:
        expected_by_path[rel] = sha
    expected_set = set(expected_by_path.keys())

    actual_set: set[str] = set()
    for p in pack_root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(pack_root).as_posix()
        if rel in {CHECKSUMS_FILENAME, SIGNATURE_FILENAME}:
            continue
        if rel in _OPTIONAL_UNCHECKSUMED_FILES:
            if rel not in expected_set:
                unchecked_files.add(rel)
            continue
        actual_set.add(rel)

    extra = sorted(actual_set - expected_set)
    for rel in extra:
        issues.append(VerifyIssue(kind="unexpected_file", path=rel))

    verified_files = 0
    for rel, expected_sha in expected_by_path.items():
        target = pack_root / rel
        if not target.exists():
            issues.append(VerifyIssue(kind="missing_file", path=rel, expected_sha256=expected_sha))
            continue
        if not target.is_file():
            issues.append(VerifyIssue(kind="io_error", path=rel, detail="not a file"))
            continue
        try:
            actual_sha = sha256_file(target)
            verified_files += 1
            if actual_sha.lower() != expected_sha.lower():
                issues.append(
                    VerifyIssue(
                        kind="mismatch",
                        path=rel,
                        expected_sha256=expected_sha,
                        actual_sha256=actual_sha,
                    )
                )
        except Exception as exc:
            issues.append(VerifyIssue(kind="io_error", path=rel, detail=str(exc)[:200]))

    manifest_path = pack_root / MANIFEST_FILENAME
    manifest_sha256 = ""
    if manifest_path.exists() and manifest_path.is_file():
        manifest_sha256 = sha256_file(manifest_path)

    signature_path = pack_root / SIGNATURE_FILENAME
    signature_doc: dict[str, Any] | None = None
    signature_present = signature_path.exists() and signature_path.is_file()
    if signature_present:
        try:
            loaded = json.loads(signature_path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(loaded, dict):
                signature_doc = loaded
            else:
                issues.append(
                    VerifyIssue(
                        kind="signature_error",
                        path=str(signature_path),
                        detail="signature.json must be an object",
                    )
                )
        except Exception as exc:
            issues.append(VerifyIssue(kind="signature_error", path=str(signature_path), detail=str(exc)[:200]))

    signature_present, signature_verified, signature_key_id, signature_key_source = _verify_signature(
        signature_present=signature_present,
        signature_doc=signature_doc,
        signature_path=str(signature_path),
        checksums_sha256=checksums_sha256,
        manifest_sha256=manifest_sha256,
        public_key=public_key,
        public_key_dir=public_key_dir,
        require_signature=require_signature,
        require_public_key=require_public_key,
        trust_embedded_public_key=trust_embedded_public_key,
        revoked_key_ids={s.strip().lower() for s in (revoked_key_ids or set()) if s.strip() != ""},
        issues=issues,
    )

    finalize_record_sha256 = ""
    finalize_report_sha256 = ""
    finalize_record_path = pack_root / FINALIZE_RECORD_REL
    if finalize_record_path.exists() and finalize_record_path.is_file():
        finalize_record_sha256 = sha256_file(finalize_record_path)
    validate_report_path = pack_root / VALIDATE_REPORT_REL
    if validate_report_path.exists() and validate_report_path.is_file():
        finalize_report_sha256 = sha256_file(validate_report_path)

    attestation_path = pack_root / ATTESTATION_FILENAME
    attestation_doc: dict[str, Any] | None = None
    attestation_present = attestation_path.exists() and attestation_path.is_file()
    if attestation_present:
        try:
            loaded = json.loads(attestation_path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(loaded, dict):
                attestation_doc = loaded
            else:
                issues.append(
                    VerifyIssue(
                        kind="attestation_error",
                        path=str(attestation_path),
                        detail="attestation.json must be an object",
                    )
                )
        except Exception as exc:
            issues.append(VerifyIssue(kind="attestation_error", path=str(attestation_path), detail=str(exc)[:200]))

    attestation_present, attestation_verified, attestation_key_id, attestation_key_source = _verify_attestation(
        attestation_present=attestation_present,
        attestation_doc=attestation_doc,
        attestation_path=str(attestation_path),
        checksums_sha256=checksums_sha256,
        finalize_record_sha256=finalize_record_sha256,
        audit_validate_report_sha256=finalize_report_sha256,
        public_key=public_key,
        public_key_dir=public_key_dir,
        require_attestation=require_attestation,
        trust_embedded_public_key=trust_embedded_public_key,
        revoked_key_ids={s.strip().lower() for s in (revoked_key_ids or set()) if s.strip() != ""},
        issues=issues,
    )

    if not strict:
        issues = [i for i in issues if i.kind != "unexpected_file"]

    return AuditVerifyResult(
        source=pack_dir,
        mode="dir",
        pack_root=str(pack_root),
        checksums_path=str(checksums_path),
        checksums_sha256=checksums_sha256,
        expected_files=len(entries),
        verified_files=verified_files,
        issues=issues,
        signature_present=signature_present,
        signature_path=str(signature_path),
        signature_verified=signature_verified,
        signature_key_id=signature_key_id,
        signature_key_source=signature_key_source,
        attestation_present=attestation_present,
        attestation_path=str(attestation_path),
        attestation_verified=attestation_verified,
        attestation_key_id=attestation_key_id,
        attestation_key_source=attestation_key_source,
        unchecked_files=sorted(unchecked_files),
    )


def _verify_zip(
    zip_path: Path,
    *,
    strict: bool = True,
    public_key: str | Path | None = None,
    public_key_dir: str | Path | None = None,
    require_signature: bool = False,
    require_public_key: bool = False,
    require_attestation: bool = False,
    trust_embedded_public_key: bool = False,
    revoked_key_ids: set[str] | None = None,
) -> AuditVerifyResult:
    zip_path = zip_path.resolve()
    issues: list[VerifyIssue] = []
    unchecked_files: set[str] = set()

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [n for n in zf.namelist() if n.endswith(CHECKSUMS_FILENAME) and not n.endswith("/")]
        if not members:
            raise FileNotFoundError(f"missing {CHECKSUMS_FILENAME} inside zip: {zip_path}")

        checksums_name = min(members, key=lambda n: (n.count("/"), len(n)))
        prefix = checksums_name[: -len(CHECKSUMS_FILENAME)]
        if prefix.endswith("/"):
            prefix = prefix[:-1]

        checksums_bytes = zf.read(checksums_name)
        checksums_sha256 = hashlib.sha256(checksums_bytes).hexdigest()
        checksums_text = checksums_bytes.decode("utf-8", errors="replace")

        entries, parse_issues = _parse_checksums_text(checksums_text, display_path=f"{zip_path}!{checksums_name}")
        issues.extend(parse_issues)

        expected_by_path: dict[str, str] = {rel: sha for sha, rel in entries}
        expected_set = set(expected_by_path.keys())

        root_prefix = f"{prefix}/" if prefix else ""
        member_set = set(zf.namelist())
        signature_name = f"{root_prefix}{SIGNATURE_FILENAME}" if root_prefix else SIGNATURE_FILENAME
        attestation_name = f"{root_prefix}{ATTESTATION_FILENAME}" if root_prefix else ATTESTATION_FILENAME
        manifest_name = f"{root_prefix}{MANIFEST_FILENAME}" if root_prefix else MANIFEST_FILENAME
        finalize_record_name = f"{root_prefix}{FINALIZE_RECORD_REL}" if root_prefix else FINALIZE_RECORD_REL
        validate_report_name = f"{root_prefix}{VALIDATE_REPORT_REL}" if root_prefix else VALIDATE_REPORT_REL

        actual_set: set[str] = set()
        for name in member_set:
            if name.endswith("/"):
                continue
            if name == checksums_name:
                continue
            if name == signature_name:
                continue
            if prefix:
                if not name.startswith(root_prefix):
                    issues.append(
                        VerifyIssue(kind="unexpected_file", path=name, detail="outside pack root")
                    )
                    continue
                rel = name[len(root_prefix):]
            else:
                rel = name
            if rel in _OPTIONAL_UNCHECKSUMED_FILES:
                if rel not in expected_set:
                    unchecked_files.add(rel)
                continue
            actual_set.add(rel)

        extra = sorted(actual_set - expected_set)
        for rel in extra:
            issues.append(VerifyIssue(kind="unexpected_file", path=rel))

        verified_files = 0
        for rel, expected_sha in expected_by_path.items():
            member_name = f"{root_prefix}{rel}" if root_prefix else rel
            if member_name not in member_set:
                issues.append(VerifyIssue(kind="missing_file", path=rel, expected_sha256=expected_sha))
                continue
            try:
                with zf.open(member_name, "r") as fp:
                    actual_sha = _sha256_stream(fp)
                verified_files += 1
                if actual_sha.lower() != expected_sha.lower():
                    issues.append(
                        VerifyIssue(
                            kind="mismatch",
                            path=rel,
                            expected_sha256=expected_sha,
                            actual_sha256=actual_sha,
                        )
                    )
            except Exception as exc:
                issues.append(VerifyIssue(kind="io_error", path=rel, detail=str(exc)[:200]))

        manifest_sha256 = ""
        if manifest_name in member_set:
            try:
                manifest_sha256 = hashlib.sha256(zf.read(manifest_name)).hexdigest()
            except Exception as exc:
                issues.append(VerifyIssue(kind="io_error", path=manifest_name, detail=str(exc)[:200]))

        signature_doc: dict[str, Any] | None = None
        signature_present = signature_name in member_set
        if signature_present:
            try:
                signature_raw = zf.read(signature_name).decode("utf-8", errors="replace")
                loaded = json.loads(signature_raw)
                if isinstance(loaded, dict):
                    signature_doc = loaded
                else:
                    issues.append(
                        VerifyIssue(
                            kind="signature_error",
                            path=f"{zip_path}!{signature_name}",
                            detail="signature.json must be an object",
                        )
                    )
            except Exception as exc:
                issues.append(
                    VerifyIssue(
                        kind="signature_error",
                        path=f"{zip_path}!{signature_name}",
                        detail=str(exc)[:200],
                    )
                )

        signature_present, signature_verified, signature_key_id, signature_key_source = _verify_signature(
            signature_present=signature_present,
            signature_doc=signature_doc,
            signature_path=f"{zip_path}!{signature_name}",
            checksums_sha256=checksums_sha256,
            manifest_sha256=manifest_sha256,
            public_key=public_key,
            public_key_dir=public_key_dir,
            require_signature=require_signature,
            require_public_key=require_public_key,
            trust_embedded_public_key=trust_embedded_public_key,
            revoked_key_ids={s.strip().lower() for s in (revoked_key_ids or set()) if s.strip() != ""},
            issues=issues,
        )

        finalize_record_sha256 = ""
        if finalize_record_name in member_set:
            try:
                finalize_record_sha256 = hashlib.sha256(zf.read(finalize_record_name)).hexdigest()
            except Exception as exc:
                issues.append(VerifyIssue(kind="io_error", path=finalize_record_name, detail=str(exc)[:200]))
        finalize_report_sha256 = ""
        if validate_report_name in member_set:
            try:
                finalize_report_sha256 = hashlib.sha256(zf.read(validate_report_name)).hexdigest()
            except Exception as exc:
                issues.append(VerifyIssue(kind="io_error", path=validate_report_name, detail=str(exc)[:200]))

        attestation_doc: dict[str, Any] | None = None
        attestation_present = attestation_name in member_set
        if attestation_present:
            try:
                attestation_raw = zf.read(attestation_name).decode("utf-8", errors="replace")
                loaded = json.loads(attestation_raw)
                if isinstance(loaded, dict):
                    attestation_doc = loaded
                else:
                    issues.append(
                        VerifyIssue(
                            kind="attestation_error",
                            path=f"{zip_path}!{attestation_name}",
                            detail="attestation.json must be an object",
                        )
                    )
            except Exception as exc:
                issues.append(
                    VerifyIssue(
                        kind="attestation_error",
                        path=f"{zip_path}!{attestation_name}",
                        detail=str(exc)[:200],
                    )
                )

        attestation_present, attestation_verified, attestation_key_id, attestation_key_source = _verify_attestation(
            attestation_present=attestation_present,
            attestation_doc=attestation_doc,
            attestation_path=f"{zip_path}!{attestation_name}",
            checksums_sha256=checksums_sha256,
            finalize_record_sha256=finalize_record_sha256,
            audit_validate_report_sha256=finalize_report_sha256,
            public_key=public_key,
            public_key_dir=public_key_dir,
            require_attestation=require_attestation,
            trust_embedded_public_key=trust_embedded_public_key,
            revoked_key_ids={s.strip().lower() for s in (revoked_key_ids or set()) if s.strip() != ""},
            issues=issues,
        )

    if not strict:
        issues = [i for i in issues if i.kind != "unexpected_file"]

    pack_root_display = prefix if prefix else "(zip root)"
    return AuditVerifyResult(
        source=zip_path,
        mode="zip",
        pack_root=pack_root_display,
        checksums_path=f"{zip_path}!{checksums_name}",
        checksums_sha256=checksums_sha256,
        expected_files=len(entries),
        verified_files=verified_files,
        issues=issues,
        signature_present=signature_present,
        signature_path=f"{zip_path}!{signature_name}",
        signature_verified=signature_verified,
        signature_key_id=signature_key_id,
        signature_key_source=signature_key_source,
        attestation_present=attestation_present,
        attestation_path=f"{zip_path}!{attestation_name}",
        attestation_verified=attestation_verified,
        attestation_key_id=attestation_key_id,
        attestation_key_source=attestation_key_source,
        unchecked_files=sorted(unchecked_files),
    )


def verify_audit_pack(
    path: str | Path,
    *,
    strict: bool = True,
    public_key: str | Path | None = None,
    public_key_dir: str | Path | None = None,
    require_signature: bool = False,
    require_public_key: bool = False,
    require_attestation: bool = False,
    trust_embedded_public_key: bool = False,
    revoked_key_ids: set[str] | None = None,
) -> AuditVerifyResult:
    """Verify an audit pack directory or zip bundle."""
    p = Path(path).expanduser()
    if p.is_dir():
        return _verify_dir(
            p,
            strict=strict,
            public_key=public_key,
            public_key_dir=public_key_dir,
            require_signature=require_signature,
            require_public_key=require_public_key,
            require_attestation=require_attestation,
            trust_embedded_public_key=trust_embedded_public_key,
            revoked_key_ids=revoked_key_ids,
        )
    if p.is_file() and p.suffix.lower() == ".zip":
        return _verify_zip(
            p,
            strict=strict,
            public_key=public_key,
            public_key_dir=public_key_dir,
            require_signature=require_signature,
            require_public_key=require_public_key,
            require_attestation=require_attestation,
            trust_embedded_public_key=trust_embedded_public_key,
            revoked_key_ids=revoked_key_ids,
        )
    raise FileNotFoundError(f"pack path not found or unsupported: {p}")
