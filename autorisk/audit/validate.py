"""Schema and semantic validation for audit packs."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)

CHECKSUMS_FILENAME = "checksums.sha256.txt"
DEFAULT_SCHEMA_RESOURCE_PACKAGE = "autorisk.resources.schemas"
VALID_SEVERITIES = {"NONE", "LOW", "MEDIUM", "HIGH"}
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_AUDIT_GRADE_REQUIRED_FILES = {
    "signature.json",
    "run_artifacts/finalize_record.json",
    "run_artifacts/audit_validate_report.json",
    "run_artifacts/policy_snapshot.json",
    "run_artifacts/review_apply_report.json",
    "run_artifacts/review_diff_report.json",
    "run_artifacts/cosmos_results_reviewed.json",
}


@dataclass
class ValidateIssue:
    kind: str  # missing_file | parse_error | schema_error | semantic_error | io_error
    path: str
    detail: str = ""
    line: int | None = None


@dataclass
class AuditValidateResult:
    source: Path
    mode: str
    pack_root: str
    schema_dir: str
    files_validated: int
    records_validated: int
    issues: list[ValidateIssue]

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "mode": self.mode,
            "pack_root": self.pack_root,
            "schema_dir": self.schema_dir,
            "files_validated": int(self.files_validated),
            "records_validated": int(self.records_validated),
            "ok": self.ok,
            "issues": [asdict(i) for i in self.issues],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


@dataclass
class _SchemaTarget:
    file_path: str
    schema_file: str
    required: bool = False
    jsonl: bool = False


_TARGETS = [
    _SchemaTarget("manifest.json", "manifest.schema.json", required=True),
    _SchemaTarget("decision_trace.jsonl", "decision_trace.schema.json", required=True, jsonl=True),
    _SchemaTarget("signature.json", "signature.schema.json", required=False),
    _SchemaTarget("attestation.json", "attestation.schema.json", required=False),
    _SchemaTarget("run_artifacts/policy_report.json", "policy_report.schema.json", required=False),
    _SchemaTarget("run_artifacts/review_queue.json", "review_queue.schema.json", required=False),
    _SchemaTarget("run_artifacts/policy_snapshot.json", "policy_snapshot.schema.json", required=False),
    _SchemaTarget("run_artifacts/audit_validate_report.json", "audit_validate_report.schema.json", required=False),
    _SchemaTarget("run_artifacts/review_apply_report.json", "review_apply_report.schema.json", required=False),
    _SchemaTarget("run_artifacts/review_diff_report.json", "review_diff_report.schema.json", required=False),
    _SchemaTarget("run_artifacts/finalize_record.json", "finalize_record.schema.json", required=False),
]


class _DirPackAccessor:
    def __init__(self, source: Path, root: Path) -> None:
        self.source = source
        self.root = root
        self.mode = "dir"
        self.pack_root = str(root)

    def exists(self, rel_path: str) -> bool:
        return (self.root / rel_path).exists()

    def read_bytes(self, rel_path: str) -> bytes:
        return (self.root / rel_path).read_bytes()

    def read_text(self, rel_path: str) -> str:
        return (self.root / rel_path).read_text(encoding="utf-8", errors="replace")

    def list_files(self) -> list[str]:
        out: list[str] = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                out.append(path.relative_to(self.root).as_posix())
        return out


class _ZipPackAccessor:
    def __init__(self, source: Path, prefix: str, checksums_name: str) -> None:
        self.source = source
        self.prefix = prefix
        self.checksums_name = checksums_name
        self.mode = "zip"
        self.pack_root = prefix if prefix else "(zip root)"

    def _member_name(self, rel_path: str) -> str:
        if self.prefix == "":
            return rel_path
        return f"{self.prefix}/{rel_path}"

    def exists(self, rel_path: str) -> bool:
        member = self._member_name(rel_path)
        with zipfile.ZipFile(self.source, "r") as zf:
            return member in set(zf.namelist())

    def read_bytes(self, rel_path: str) -> bytes:
        member = self._member_name(rel_path)
        with zipfile.ZipFile(self.source, "r") as zf:
            return zf.read(member)

    def read_text(self, rel_path: str) -> str:
        return self.read_bytes(rel_path).decode("utf-8", errors="replace")

    def list_files(self) -> list[str]:
        out: list[str] = []
        root_prefix = f"{self.prefix}/" if self.prefix else ""
        with zipfile.ZipFile(self.source, "r") as zf:
            for name in sorted(set(zf.namelist())):
                if name.endswith("/"):
                    continue
                if self.prefix:
                    if not name.startswith(root_prefix):
                        continue
                    rel = name[len(root_prefix):]
                else:
                    rel = name
                out.append(rel)
        return out


def _load_schema(schema_file: str, *, schema_dir: Path | None) -> dict[str, Any]:
    if schema_dir is not None:
        path = schema_dir / schema_file
        if not path.exists():
            raise FileNotFoundError(f"schema file not found: {path}")
        text = path.read_text(encoding="utf-8")
    else:
        resource = importlib_resources.files(DEFAULT_SCHEMA_RESOURCE_PACKAGE).joinpath(schema_file)
        text = resource.read_text(encoding="utf-8")

    payload = json.loads(text)
    if not isinstance(payload, dict):
        if schema_dir is not None:
            raise ValueError(f"schema must be JSON object: {schema_dir / schema_file}")
        raise ValueError(f"schema must be JSON object: package://{DEFAULT_SCHEMA_RESOURCE_PACKAGE}/{schema_file}")
    return payload


def _schema_errors(validator: Draft202012Validator, instance: Any, path: str) -> list[ValidateIssue]:
    issues: list[ValidateIssue] = []
    for err in sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path)):
        at = ".".join(str(x) for x in err.absolute_path)
        detail = err.message
        if at != "":
            detail = f"{at}: {detail}"
        issues.append(ValidateIssue(kind="schema_error", path=path, detail=detail[:300]))
    return issues


def _normalize_rank(raw: Any) -> int | None:
    try:
        return int(raw)
    except Exception:
        return None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_record_sha(record: dict[str, Any]) -> str:
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_hex64(text: str) -> bool:
    return bool(_HEX64.match(str(text).strip()))


def _parse_jsonl_objects(text: str, *, path: str) -> tuple[list[tuple[int, dict[str, Any]]], list[ValidateIssue]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    issues: list[ValidateIssue] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if line == "" or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except Exception as exc:
            issues.append(
                ValidateIssue(
                    kind="parse_error",
                    path=path,
                    line=lineno,
                    detail=str(exc)[:200],
                )
            )
            continue
        if not isinstance(obj, dict):
            issues.append(
                ValidateIssue(
                    kind="parse_error",
                    path=path,
                    line=lineno,
                    detail="jsonl row must be an object",
                )
            )
            continue
        rows.append((lineno, obj))
    return rows, issues


def _required_files_for_profile(
    *,
    profile: str,
    require_signature: bool = False,
    require_finalize_record: bool = False,
    require_validate_report: bool = False,
    require_policy_snapshot: bool = False,
    require_review_artifacts: bool = False,
) -> set[str]:
    required: set[str] = set()
    normalized_profile = str(profile or "default").strip().lower()
    if normalized_profile == "audit-grade":
        required.update(_AUDIT_GRADE_REQUIRED_FILES)

    if require_signature:
        required.add("signature.json")
    if require_finalize_record:
        required.add("run_artifacts/finalize_record.json")
    if require_validate_report:
        required.add("run_artifacts/audit_validate_report.json")
    if require_policy_snapshot:
        required.add("run_artifacts/policy_snapshot.json")
    if require_review_artifacts:
        required.update(
            {
                "run_artifacts/review_apply_report.json",
                "run_artifacts/review_diff_report.json",
                "run_artifacts/cosmos_results_reviewed.json",
            }
        )
    return required


def _append_missing_required_file_issues(
    *,
    accessor: _DirPackAccessor | _ZipPackAccessor,
    required_paths: set[str],
    issues: list[ValidateIssue],
) -> None:
    already_missing = {
        issue.path
        for issue in issues
        if issue.kind == "missing_file"
    }
    for rel in sorted(required_paths):
        if rel in already_missing:
            continue
        if not accessor.exists(rel):
            issues.append(
                ValidateIssue(
                    kind="missing_file",
                    path=rel,
                    detail="required by validation profile",
                )
            )


def _semantic_check_trace(
    trace_rows: list[tuple[int, dict[str, Any]]],
    *,
    path: str,
) -> list[ValidateIssue]:
    issues: list[ValidateIssue] = []
    seen_rank_lines: dict[int, int] = {}

    for line, row in trace_rows:
        rank = _normalize_rank(row.get("candidate_rank"))
        if rank is None:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path=path,
                    line=line,
                    detail="candidate_rank must be integer",
                )
            )
            continue
        if rank in seen_rank_lines:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path=path,
                    line=line,
                    detail=f"duplicate candidate_rank={rank} (first at line {seen_rank_lines[rank]})",
                )
            )
        else:
            seen_rank_lines[rank] = line
            if rank <= 0:
                issues.append(
                    ValidateIssue(
                        kind="semantic_error",
                        path=path,
                        line=line,
                        detail=f"candidate_rank must be positive: {rank}",
                    )
                )

        final_assessment = row.get("final_assessment", {})
        if isinstance(final_assessment, dict):
            severity = str(final_assessment.get("severity", "")).upper()
            if severity not in VALID_SEVERITIES:
                issues.append(
                    ValidateIssue(
                        kind="semantic_error",
                        path=path,
                        line=line,
                        detail=f"invalid severity: {severity}",
                    )
                )
            confidence_raw = final_assessment.get("confidence")
            try:
                confidence = float(confidence_raw)
                if confidence < 0.0 or confidence > 1.0:
                    issues.append(
                        ValidateIssue(
                            kind="semantic_error",
                            path=path,
                            line=line,
                            detail=f"confidence out of range [0,1]: {confidence}",
                        )
                    )
            except Exception:
                pass

            parse_success_recorded = bool((row.get("parsing", {}) or {}).get("parse_success_recorded", True))
            error_text = str(final_assessment.get("error", "")).strip()
            if (not parse_success_recorded) and error_text == "":
                issues.append(
                    ValidateIssue(
                        kind="semantic_error",
                        path=path,
                        line=line,
                        detail="parse_success_recorded=false requires non-empty final_assessment.error",
                    )
                )

    if seen_rank_lines:
        ranks = sorted(seen_rank_lines.keys())
        if ranks[0] != 1:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path=path,
                    detail=f"candidate_rank should start at 1 (found min={ranks[0]})",
                )
            )
        expected = set(range(ranks[0], ranks[-1] + 1))
        missing = sorted(expected - set(ranks))
        if missing:
            preview = ",".join(str(x) for x in missing[:20])
            tail = "" if len(missing) <= 20 else f" (+{len(missing) - 20} more)"
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path=path,
                    detail=f"missing candidate_rank values: {preview}{tail}",
                )
            )

    return issues


def _semantic_check_manifest(
    manifest_obj: dict[str, Any] | None,
    trace_rows: list[tuple[int, dict[str, Any]]],
) -> list[ValidateIssue]:
    issues: list[ValidateIssue] = []
    if manifest_obj is None:
        return issues

    summary = manifest_obj.get("summary", {})
    if not isinstance(summary, dict):
        return issues

    records = _normalize_rank(summary.get("records"))
    if records is not None and records != len(trace_rows):
        issues.append(
            ValidateIssue(
                kind="semantic_error",
                path="manifest.json",
                detail=f"summary.records={records} but decision_trace rows={len(trace_rows)}",
            )
        )

    recorded_parse_success = _normalize_rank(summary.get("parse_success_recorded"))
    if recorded_parse_success is not None:
        actual = sum(1 for _, r in trace_rows if bool((r.get("parsing", {}) or {}).get("parse_success_recorded", False)))
        if recorded_parse_success != actual:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path="manifest.json",
                    detail=f"summary.parse_success_recorded={recorded_parse_success} but actual={actual}",
                )
            )

    hist = summary.get("severity_histogram", {})
    if isinstance(hist, dict) and trace_rows:
        actual_hist = {"NONE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for _, row in trace_rows:
            severity = str(((row.get("final_assessment", {}) or {}).get("severity", "NONE"))).upper()
            if severity in actual_hist:
                actual_hist[severity] += 1
            else:
                actual_hist[severity] = actual_hist.get(severity, 0) + 1
        for sev in ["NONE", "LOW", "MEDIUM", "HIGH"]:
            expected = _normalize_rank(hist.get(sev))
            if expected is not None and expected != actual_hist.get(sev, 0):
                issues.append(
                    ValidateIssue(
                        kind="semantic_error",
                        path="manifest.json",
                        detail=f"severity_histogram[{sev}]={expected} but actual={actual_hist.get(sev, 0)}",
                    )
                )
    return issues


def _semantic_check_results_json(results_obj: Any, *, path: str) -> list[ValidateIssue]:
    issues: list[ValidateIssue] = []
    if not isinstance(results_obj, list):
        return issues

    seen_rank: dict[int, int] = {}
    for idx, row in enumerate(results_obj, start=1):
        if not isinstance(row, dict):
            continue
        rank = _normalize_rank(row.get("candidate_rank"))
        if rank is None:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path=path,
                    detail=f"entry {idx}: candidate_rank must be integer",
                )
            )
            continue
        if rank in seen_rank:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path=path,
                    detail=f"entry {idx}: duplicate candidate_rank={rank} (first at entry {seen_rank[rank]})",
                )
            )
        else:
            seen_rank[rank] = idx
            if rank <= 0:
                issues.append(
                    ValidateIssue(
                        kind="semantic_error",
                        path=path,
                        detail=f"entry {idx}: candidate_rank must be positive",
                    )
                )

        severity = str(row.get("severity", "")).upper()
        if severity not in VALID_SEVERITIES:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path=path,
                    detail=f"entry {idx}: invalid severity={severity}",
                )
            )
        parse_success = bool(row.get("parse_success", True))
        error_text = str(row.get("error", "")).strip()
        if (not parse_success) and error_text == "":
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path=path,
                    detail=f"entry {idx}: parse_success=false requires non-empty error",
                )
            )

    if seen_rank:
        ranks = sorted(seen_rank.keys())
        if ranks[0] != 1:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path=path,
                    detail=f"candidate_rank should start at 1 (found min={ranks[0]})",
                )
            )
        expected = set(range(ranks[0], ranks[-1] + 1))
        missing = sorted(expected - set(ranks))
        if missing:
            preview = ",".join(str(x) for x in missing[:20])
            tail = "" if len(missing) <= 20 else f" (+{len(missing) - 20} more)"
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path=path,
                    detail=f"missing candidate_rank values: {preview}{tail}",
                )
            )
    return issues


def _semantic_check_policy_and_queue(
    policy_obj: dict[str, Any] | None,
    queue_obj: dict[str, Any] | None,
) -> list[ValidateIssue]:
    issues: list[ValidateIssue] = []
    if policy_obj is None or queue_obj is None:
        return issues

    policy_summary = policy_obj.get("summary", {})
    queue_summary = queue_obj.get("summary", {})
    if not isinstance(policy_summary, dict) or not isinstance(queue_summary, dict):
        return issues

    required_review = _normalize_rank(policy_summary.get("required_review_count"))
    queue_required = _normalize_rank(queue_summary.get("required_review_count"))
    queue_size = _normalize_rank(queue_summary.get("queue_size"))
    missing_review = _normalize_rank(policy_summary.get("missing_review_count"))

    if required_review is not None and queue_required is not None and required_review != queue_required:
        issues.append(
            ValidateIssue(
                kind="semantic_error",
                path="run_artifacts/policy_report.json",
                detail=f"required_review_count mismatch policy={required_review} queue={queue_required}",
            )
        )
    if missing_review is not None and queue_size is not None and missing_review != queue_size:
        issues.append(
            ValidateIssue(
                kind="semantic_error",
                path="run_artifacts/policy_report.json",
                detail=f"missing_review_count={missing_review} but queue_size={queue_size}",
            )
        )
    return issues


def _semantic_check_policy_snapshot(
    policy_obj: dict[str, Any] | None,
    snapshot_obj: dict[str, Any] | None,
) -> list[ValidateIssue]:
    issues: list[ValidateIssue] = []
    if policy_obj is None or snapshot_obj is None:
        return issues

    policy_source = policy_obj.get("policy_source", {})
    snapshot_source = snapshot_obj.get("policy_source", {})
    if isinstance(policy_source, dict) and isinstance(snapshot_source, dict):
        p_sha = str(policy_source.get("policy_sha256", "")).strip()
        s_sha = str(snapshot_source.get("policy_sha256", "")).strip()
        if p_sha != "" and s_sha != "" and p_sha != s_sha:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path="run_artifacts/policy_snapshot.json",
                    detail="policy_source.policy_sha256 mismatch vs policy_report",
                )
            )
    policy_policy = policy_obj.get("policy", {})
    snapshot_policy = snapshot_obj.get("policy", {})
    if isinstance(policy_policy, dict) and isinstance(snapshot_policy, dict):
        if policy_policy != snapshot_policy:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path="run_artifacts/policy_snapshot.json",
                    detail="policy contents mismatch vs policy_report.policy",
                )
            )
    return issues


def _semantic_check_review_chain(
    reviewed_results_obj: Any,
    review_log_rows: list[tuple[int, dict[str, Any]]],
) -> list[ValidateIssue]:
    issues: list[ValidateIssue] = []
    if not isinstance(reviewed_results_obj, list):
        return issues
    if not review_log_rows:
        return issues

    review_sha_set: set[str] = set()
    for _, row in review_log_rows:
        review_sha_set.add(_canonical_record_sha(row))

    for idx, row in enumerate(reviewed_results_obj, start=1):
        if not isinstance(row, dict):
            continue
        review_meta = row.get("review")
        if not isinstance(review_meta, dict):
            continue
        record_sha = str(review_meta.get("record_sha256", "")).strip().lower()
        if record_sha == "":
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path="run_artifacts/cosmos_results_reviewed.json",
                    detail=f"entry {idx}: missing review.record_sha256",
                )
            )
            continue
        if record_sha not in review_sha_set:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path="run_artifacts/cosmos_results_reviewed.json",
                    detail=f"entry {idx}: review.record_sha256 not found in review_log",
                )
            )
    return issues


def _semantic_check_review_reports(
    apply_obj: dict[str, Any] | None,
    diff_obj: dict[str, Any] | None,
) -> list[ValidateIssue]:
    issues: list[ValidateIssue] = []
    if apply_obj is None or diff_obj is None:
        return issues

    apply_count = _normalize_rank(apply_obj.get("applied"))
    diff_count = _normalize_rank(diff_obj.get("applied"))
    if apply_count is not None and diff_count is not None and apply_count != diff_count:
        issues.append(
            ValidateIssue(
                kind="semantic_error",
                path="run_artifacts/review_apply_report.json",
                detail=f"applied mismatch vs review_diff_report: {apply_count} != {diff_count}",
            )
        )

    override_count = _normalize_rank(diff_obj.get("override_count"))
    changed_rows = diff_obj.get("changed_candidates", [])
    if override_count is not None and isinstance(changed_rows, list) and override_count != len(changed_rows):
        issues.append(
            ValidateIssue(
                kind="semantic_error",
                path="run_artifacts/review_diff_report.json",
                detail=f"override_count={override_count} but changed_candidates={len(changed_rows)}",
            )
        )
    return issues


def _semantic_check_finalize_record(
    finalize_obj: dict[str, Any] | None,
    *,
    checksums_sha256: str,
    signature_obj: dict[str, Any] | None = None,
    validate_report_obj: dict[str, Any] | None = None,
    pack_has_validate_report: bool = False,
) -> list[ValidateIssue]:
    issues: list[ValidateIssue] = []
    if finalize_obj is None:
        return issues

    pack_fingerprint = str(finalize_obj.get("pack_fingerprint", "")).strip().lower()
    if pack_fingerprint != "" and pack_fingerprint != checksums_sha256.lower():
        issues.append(
            ValidateIssue(
                kind="semantic_error",
                path="run_artifacts/finalize_record.json",
                detail="pack_fingerprint does not match checksums.sha256.txt hash",
            )
        )
    signature_present = bool(finalize_obj.get("signature_present", False))
    signature_key_id = str(finalize_obj.get("signature_key_id", "")).strip()
    if signature_present and signature_key_id == "":
        issues.append(
            ValidateIssue(
                kind="semantic_error",
                path="run_artifacts/finalize_record.json",
                detail="signature_present=true requires non-empty signature_key_id",
            )
        )
    if signature_obj is not None:
        signed = signature_obj.get("signed", {})
        signed_key_id = str((signed or {}).get("key_id", "")).strip()
        if signature_key_id != "" and signed_key_id != "" and signature_key_id != signed_key_id:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path="run_artifacts/finalize_record.json",
                    detail="signature_key_id mismatch vs signature.json signed.key_id",
                )
            )
        if signature_present is not True:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path="run_artifacts/finalize_record.json",
                    detail="signature.json exists but finalize_record.signature_present=false",
                )
            )
    else:
        if signature_present:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path="run_artifacts/finalize_record.json",
                    detail="finalize_record.signature_present=true but signature.json is missing",
                )
            )

    audit_grade = bool(finalize_obj.get("audit_grade", False))
    require_signature = bool(finalize_obj.get("require_signature", False))
    require_trusted_key = bool(finalize_obj.get("require_trusted_key", False))
    trust_embedded = bool(finalize_obj.get("trust_embedded_public_key", False))
    if audit_grade:
        if not require_signature:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path="run_artifacts/finalize_record.json",
                    detail="audit_grade=true requires require_signature=true",
                )
            )
        if not require_trusted_key:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path="run_artifacts/finalize_record.json",
                    detail="audit_grade=true requires require_trusted_key=true",
                )
            )
        if trust_embedded:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path="run_artifacts/finalize_record.json",
                    detail="audit_grade=true requires trust_embedded_public_key=false",
                )
            )
    if require_signature and not signature_present:
        issues.append(
            ValidateIssue(
                kind="semantic_error",
                path="run_artifacts/finalize_record.json",
                detail="require_signature=true but signature_present=false",
            )
        )
    policy_source = finalize_obj.get("policy_source", {})
    policy_sha = str(finalize_obj.get("policy_sha256", "")).strip()
    if isinstance(policy_source, dict):
        policy_source_sha = str(policy_source.get("policy_sha256", "")).strip()
        if policy_sha != "" and policy_source_sha != "" and policy_sha != policy_source_sha:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path="run_artifacts/finalize_record.json",
                    detail="policy_sha256 does not match policy_source.policy_sha256",
            )
        )

    validate_ok_raw = finalize_obj.get("validate_ok")
    validate_issues_count = _normalize_rank(finalize_obj.get("validate_issues_count"))
    validate_report_path = str(finalize_obj.get("validate_report_path", "")).strip()
    if isinstance(validate_ok_raw, bool) and validate_issues_count is not None:
        if validate_ok_raw and validate_issues_count != 0:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path="run_artifacts/finalize_record.json",
                    detail="validate_ok=true requires validate_issues_count=0",
                )
            )
        if (not validate_ok_raw) and validate_issues_count == 0:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path="run_artifacts/finalize_record.json",
                    detail="validate_ok=false should have validate_issues_count>0",
                )
            )
    if validate_report_path != "":
        normalized_validate_path = validate_report_path.replace("\\", "/")
        if normalized_validate_path != "run_artifacts/audit_validate_report.json":
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path="run_artifacts/finalize_record.json",
                    detail="validate_report_path should be run_artifacts/audit_validate_report.json",
                )
            )
        if not pack_has_validate_report:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path="run_artifacts/finalize_record.json",
                    detail="validate_report_path is set but run_artifacts/audit_validate_report.json is missing",
                )
            )
    if isinstance(validate_report_obj, dict):
        report_ok = validate_report_obj.get("ok")
        report_issues = validate_report_obj.get("issues", [])
        report_issue_count = len(report_issues) if isinstance(report_issues, list) else None
        if isinstance(validate_ok_raw, bool) and isinstance(report_ok, bool) and validate_ok_raw != report_ok:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path="run_artifacts/finalize_record.json",
                    detail="validate_ok mismatch vs audit_validate_report.ok",
                )
            )
        if validate_issues_count is not None and report_issue_count is not None and validate_issues_count != report_issue_count:
            issues.append(
                ValidateIssue(
                    kind="semantic_error",
                    path="run_artifacts/finalize_record.json",
                    detail=f"validate_issues_count mismatch vs audit_validate_report: {validate_issues_count} != {report_issue_count}",
                )
            )

    handoff_path = str(finalize_obj.get("handoff_path", "")).strip()
    handoff_checksums_sha256 = str(finalize_obj.get("handoff_checksums_sha256", "")).strip()
    handoff_pack_zip_sha256 = str(finalize_obj.get("handoff_pack_zip_sha256", "")).strip()
    handoff_bundle_zip_sha256 = str(finalize_obj.get("handoff_verifier_bundle_zip_sha256", "")).strip()
    handoff_anchor_checksums_sha256 = str(finalize_obj.get("handoff_anchor_checksums_sha256", "")).strip()
    handoff_anchor_bundle_sha256 = str(finalize_obj.get("handoff_anchor_verifier_bundle_zip_sha256", "")).strip()

    if any(
        value != ""
        for value in [
            handoff_path,
            handoff_checksums_sha256,
            handoff_pack_zip_sha256,
            handoff_bundle_zip_sha256,
        ]
    ):
        issues.append(
            ValidateIssue(
                kind="semantic_error",
                path="run_artifacts/finalize_record.json",
                detail=(
                    "PACK-internal run_artifacts/finalize_record.json must not include handoff_* hashes; "
                    "use handoff_anchor_* to avoid circular dependency"
                ),
            )
        )

    if handoff_anchor_checksums_sha256 != "" and not _is_hex64(handoff_anchor_checksums_sha256):
        issues.append(
            ValidateIssue(
                kind="semantic_error",
                path="run_artifacts/finalize_record.json",
                detail="handoff_anchor_checksums_sha256 must be empty or 64-hex",
            )
        )
    if handoff_anchor_bundle_sha256 != "" and not _is_hex64(handoff_anchor_bundle_sha256):
        issues.append(
            ValidateIssue(
                kind="semantic_error",
                path="run_artifacts/finalize_record.json",
                detail="handoff_anchor_verifier_bundle_zip_sha256 must be empty or 64-hex",
            )
        )
    if (handoff_anchor_checksums_sha256 == "") != (handoff_anchor_bundle_sha256 == ""):
        issues.append(
            ValidateIssue(
                kind="semantic_error",
                path="run_artifacts/finalize_record.json",
                detail="handoff_anchor_* fields must be provided together",
            )
        )
    return issues


def _resolve_dir_accessor(source: Path) -> _DirPackAccessor:
    candidates = sorted(source.rglob(CHECKSUMS_FILENAME), key=lambda p: (len(p.parts), str(p)))
    if not candidates:
        raise FileNotFoundError(f"missing {CHECKSUMS_FILENAME} under: {source}")
    checksums_path = candidates[0]
    return _DirPackAccessor(source=source, root=checksums_path.parent)


def _resolve_zip_accessor(source: Path) -> _ZipPackAccessor:
    with zipfile.ZipFile(source, "r") as zf:
        members = [n for n in zf.namelist() if n.endswith(CHECKSUMS_FILENAME) and not n.endswith("/")]
        if not members:
            raise FileNotFoundError(f"missing {CHECKSUMS_FILENAME} inside zip: {source}")
        checksums_name = min(members, key=lambda n: (n.count("/"), len(n)))
    prefix = checksums_name[: -len(CHECKSUMS_FILENAME)].rstrip("/")
    return _ZipPackAccessor(source=source, prefix=prefix, checksums_name=checksums_name)


def validate_audit_pack(
    path: str | Path,
    *,
    schema_dir: str | Path | None = None,
    semantic_checks: bool = True,
    profile: str = "default",
    require_signature: bool = False,
    require_finalize_record: bool = False,
    require_validate_report: bool = False,
    require_policy_snapshot: bool = False,
    require_review_artifacts: bool = False,
) -> AuditValidateResult:
    """Validate audit pack against schemas and semantic constraints."""
    source = Path(path).expanduser()
    schema_dir_path: Path | None = None
    schema_source: str
    if schema_dir is not None and str(schema_dir).strip() != "":
        schema_dir_path = Path(schema_dir).expanduser().resolve()
        schema_source = str(schema_dir_path)
    else:
        schema_source = f"package://{DEFAULT_SCHEMA_RESOURCE_PACKAGE}"

    if source.is_dir():
        accessor: _DirPackAccessor | _ZipPackAccessor = _resolve_dir_accessor(source.resolve())
    elif source.is_file() and source.suffix.lower() == ".zip":
        accessor = _resolve_zip_accessor(source.resolve())
    else:
        raise FileNotFoundError(f"pack path not found or unsupported: {source}")

    issues: list[ValidateIssue] = []
    files_validated = 0
    records_validated = 0
    parsed_json: dict[str, Any] = {}
    parsed_jsonl: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    trace_rows: list[tuple[int, dict[str, Any]]] = []

    for target in _TARGETS:
        if not accessor.exists(target.file_path):
            if target.required:
                issues.append(
                    ValidateIssue(kind="missing_file", path=target.file_path, detail="required file is missing")
                )
            continue

        try:
            schema = _load_schema(target.schema_file, schema_dir=schema_dir_path)
            validator = Draft202012Validator(schema)
            payload_text = accessor.read_text(target.file_path)
        except Exception as exc:
            issues.append(ValidateIssue(kind="io_error", path=target.file_path, detail=str(exc)[:300]))
            continue

        if target.jsonl:
            rows, parse_issues = _parse_jsonl_objects(payload_text, path=target.file_path)
            issues.extend(parse_issues)
            for line, obj in rows:
                records_validated += 1
                schema_issues = _schema_errors(validator, obj, target.file_path)
                for issue in schema_issues:
                    issue.line = line
                issues.extend(schema_issues)
            parsed_jsonl[target.file_path] = rows
            trace_rows = rows
            files_validated += 1
            continue

        try:
            obj = json.loads(payload_text)
        except Exception as exc:
            issues.append(ValidateIssue(kind="parse_error", path=target.file_path, detail=str(exc)[:300]))
            files_validated += 1
            continue

        parsed_json[target.file_path] = obj
        records_validated += 1
        issues.extend(_schema_errors(validator, obj, target.file_path))
        files_validated += 1

    # Validate optional human review logs if bundled (human_review/*.jsonl)
    review_log_validator: Draft202012Validator | None = None
    try:
        review_log_schema = _load_schema("review_log_entry.schema.json", schema_dir=schema_dir_path)
        review_log_validator = Draft202012Validator(review_log_schema)
    except Exception as exc:
        issues.append(
            ValidateIssue(
                kind="io_error",
                path="schemas/review_log_entry.schema.json",
                detail=str(exc)[:300],
            )
        )

    if review_log_validator is not None:
        for rel_path in accessor.list_files():
            if not rel_path.startswith("human_review/") or not rel_path.endswith(".jsonl"):
                continue
            try:
                rows, parse_issues = _parse_jsonl_objects(accessor.read_text(rel_path), path=rel_path)
                issues.extend(parse_issues)
                parsed_jsonl[rel_path] = rows
                for line, obj in rows:
                    records_validated += 1
                    schema_issues = _schema_errors(review_log_validator, obj, rel_path)
                    for issue in schema_issues:
                        issue.line = line
                    issues.extend(schema_issues)
                files_validated += 1
            except Exception as exc:
                issues.append(ValidateIssue(kind="io_error", path=rel_path, detail=str(exc)[:300]))

    required_files = _required_files_for_profile(
        profile=profile,
        require_signature=require_signature,
        require_finalize_record=require_finalize_record,
        require_validate_report=require_validate_report,
        require_policy_snapshot=require_policy_snapshot,
        require_review_artifacts=require_review_artifacts,
    )
    _append_missing_required_file_issues(
        accessor=accessor,
        required_paths=required_files,
        issues=issues,
    )

    if semantic_checks:
        issues.extend(_semantic_check_trace(trace_rows, path="decision_trace.jsonl"))
        manifest_obj = parsed_json.get("manifest.json")
        if isinstance(manifest_obj, dict):
            issues.extend(_semantic_check_manifest(manifest_obj, trace_rows))

        reviewed_results_obj: Any = None
        for results_rel in [
            "run_artifacts/cosmos_results.json",
            "run_artifacts/cosmos_results_reviewed.json",
        ]:
            if accessor.exists(results_rel):
                try:
                    results_obj = json.loads(accessor.read_text(results_rel))
                    issues.extend(_semantic_check_results_json(results_obj, path=results_rel))
                    if results_rel == "run_artifacts/cosmos_results_reviewed.json":
                        reviewed_results_obj = results_obj
                except Exception as exc:
                    issues.append(ValidateIssue(kind="parse_error", path=results_rel, detail=str(exc)[:300]))

        policy_obj = parsed_json.get("run_artifacts/policy_report.json")
        queue_obj = parsed_json.get("run_artifacts/review_queue.json")
        snapshot_obj = parsed_json.get("run_artifacts/policy_snapshot.json")
        if isinstance(policy_obj, dict) and isinstance(queue_obj, dict):
            issues.extend(_semantic_check_policy_and_queue(policy_obj, queue_obj))
        if isinstance(policy_obj, dict) and isinstance(snapshot_obj, dict):
            issues.extend(_semantic_check_policy_snapshot(policy_obj, snapshot_obj))

        apply_obj = parsed_json.get("run_artifacts/review_apply_report.json")
        diff_obj = parsed_json.get("run_artifacts/review_diff_report.json")
        if isinstance(apply_obj, dict) and isinstance(diff_obj, dict):
            issues.extend(_semantic_check_review_reports(apply_obj, diff_obj))

        all_review_rows: list[tuple[int, dict[str, Any]]] = []
        for rel_path, rows in parsed_jsonl.items():
            if rel_path.startswith("human_review/"):
                all_review_rows.extend(rows)
        if isinstance(reviewed_results_obj, list) and all_review_rows:
            issues.extend(_semantic_check_review_chain(reviewed_results_obj, all_review_rows))

        finalize_obj = parsed_json.get("run_artifacts/finalize_record.json")
        signature_obj = parsed_json.get("signature.json")
        validate_report_obj = parsed_json.get("run_artifacts/audit_validate_report.json")
        if isinstance(finalize_obj, dict):
            try:
                checksums_sha256 = _sha256_bytes(accessor.read_bytes("checksums.sha256.txt"))
                issues.extend(
                    _semantic_check_finalize_record(
                        finalize_obj,
                        checksums_sha256=checksums_sha256,
                        signature_obj=signature_obj if isinstance(signature_obj, dict) else None,
                        validate_report_obj=validate_report_obj if isinstance(validate_report_obj, dict) else None,
                        pack_has_validate_report=accessor.exists("run_artifacts/audit_validate_report.json"),
                    )
                )
            except Exception as exc:
                issues.append(ValidateIssue(kind="io_error", path="checksums.sha256.txt", detail=str(exc)[:300]))

    result = AuditValidateResult(
        source=source.resolve(),
        mode=accessor.mode,
        pack_root=accessor.pack_root,
        schema_dir=schema_source,
        files_validated=files_validated,
        records_validated=records_validated,
        issues=issues,
    )
    log.info(
        "Audit validate: source=%s mode=%s files=%d records=%d issues=%d",
        result.source,
        result.mode,
        result.files_validated,
        result.records_validated,
        len(result.issues),
    )
    return result
