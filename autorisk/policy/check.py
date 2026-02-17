from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from autorisk.review.log import DEFAULT_REVIEW_LOG_NAME
from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)

POLICY_SCHEMA_VERSION = 1
DEFAULT_POLICY_REPORT_NAME = "policy_report.json"
DEFAULT_REVIEW_QUEUE_NAME = "review_queue.json"
DEFAULT_POLICY_SNAPSHOT_NAME = "policy_snapshot.json"
REPO_POLICY_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "policy.yaml"
PACKAGED_POLICY_RESOURCE = "autorisk.resources.configs"
PACKAGED_POLICY_NAME = "policy.yaml"
DEFAULT_POLICY_CONFIG_PATH = REPO_POLICY_CONFIG_PATH
DEFAULT_REVIEW_SEVERITIES = {"MEDIUM", "HIGH"}
SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _default_policy() -> dict[str, Any]:
    return {
        "required_review_severities": sorted(DEFAULT_REVIEW_SEVERITIES),
        "require_parse_failure_review": True,
        "require_error_review": True,
        "allow_stale": False,
    }


def _load_policy_from_path(path: Path) -> dict[str, Any]:
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"policy config must be a mapping: {path}")
    return loaded


def _load_packaged_policy() -> tuple[dict[str, Any], dict[str, Any]]:
    resource = importlib_resources.files(PACKAGED_POLICY_RESOURCE).joinpath(PACKAGED_POLICY_NAME)
    text = resource.read_text(encoding="utf-8")
    with importlib_resources.as_file(resource) as resolved_path:
        loaded = _load_policy_from_path(resolved_path)
    source = {
        "source_type": "package_resource",
        "policy_path": f"package://{PACKAGED_POLICY_RESOURCE}/{PACKAGED_POLICY_NAME}",
        "policy_sha256": _sha256_text(text),
    }
    return loaded, source


def resolve_policy(
    *,
    policy_path: str | Path | None,
    required_review_severities: set[str] | None,
    require_parse_failure_review: bool | None,
    require_error_review: bool | None,
    allow_stale: bool | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    effective = _default_policy()
    source: dict[str, Any] = {
        "source_type": "default",
        "policy_path": "",
        "policy_sha256": "",
    }

    configured_path: Path | None = None
    packaged_loaded: dict[str, Any] | None = None
    if policy_path is not None and str(policy_path).strip() != "":
        configured_path = Path(policy_path).expanduser().resolve()
        if not configured_path.exists():
            raise FileNotFoundError(f"policy file not found: {configured_path}")
    elif REPO_POLICY_CONFIG_PATH.exists():
        configured_path = REPO_POLICY_CONFIG_PATH.resolve()
    else:
        try:
            packaged_loaded, source = _load_packaged_policy()
        except Exception:
            packaged_loaded = None

    if configured_path is not None:
        loaded = _load_policy_from_path(configured_path)
        raw_sev = loaded.get("required_review_severities")
        if isinstance(raw_sev, list):
            effective["required_review_severities"] = [
                str(s).upper() for s in raw_sev if str(s).strip() != ""
            ]
        for key in ["require_parse_failure_review", "require_error_review", "allow_stale"]:
            if key in loaded:
                effective[key] = bool(loaded.get(key))
        source = {
            "source_type": "file",
            "policy_path": str(configured_path),
            "policy_sha256": _sha256_file(configured_path),
        }
    elif packaged_loaded is not None:
        raw_sev = packaged_loaded.get("required_review_severities")
        if isinstance(raw_sev, list):
            effective["required_review_severities"] = [
                str(s).upper() for s in raw_sev if str(s).strip() != ""
            ]
        for key in ["require_parse_failure_review", "require_error_review", "allow_stale"]:
            if key in packaged_loaded:
                effective[key] = bool(packaged_loaded.get(key))

    if required_review_severities is not None:
        effective["required_review_severities"] = sorted({str(s).upper() for s in required_review_severities})
    if require_parse_failure_review is not None:
        effective["require_parse_failure_review"] = bool(require_parse_failure_review)
    if require_error_review is not None:
        effective["require_error_review"] = bool(require_error_review)
    if allow_stale is not None:
        effective["allow_stale"] = bool(allow_stale)

    return effective, source


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            raw = line.strip()
            if raw == "" or raw.startswith("#"):
                continue
            try:
                obj = json.loads(raw)
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {path} line {lineno}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _normalize_severity(raw: Any) -> str:
    text = str(raw or "").strip().upper()
    if text in {"NONE", "LOW", "MEDIUM", "HIGH"}:
        return text
    return "NONE"


@dataclass
class PolicyCheckResult:
    run_dir: Path
    results_path: Path
    review_log_path: Path
    report_path: Path
    queue_path: Path
    snapshot_path: Path
    policy_source: dict[str, Any]
    passed: bool
    total_candidates: int
    required_review_count: int
    reviewed_count_valid: int
    reviewed_count_stale: int
    missing_review_count: int
    violations: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "run_dir": str(self.run_dir),
            "results_path": str(self.results_path),
            "review_log_path": str(self.review_log_path),
            "report_path": str(self.report_path),
            "queue_path": str(self.queue_path),
            "snapshot_path": str(self.snapshot_path),
            "policy_source": dict(self.policy_source),
            "passed": bool(self.passed),
            "total_candidates": int(self.total_candidates),
            "required_review_count": int(self.required_review_count),
            "reviewed_count_valid": int(self.reviewed_count_valid),
            "reviewed_count_stale": int(self.reviewed_count_stale),
            "missing_review_count": int(self.missing_review_count),
            "violations": list(self.violations),
        }


def run_policy_check(
    *,
    run_dir: str | Path,
    policy_path: str | Path | None = None,
    review_log: str | Path | None = None,
    report_path: str | Path | None = None,
    queue_path: str | Path | None = None,
    snapshot_path: str | Path | None = None,
    allow_stale: bool | None = None,
    required_review_severities: set[str] | None = None,
    require_parse_failure_review: bool | None = None,
    require_error_review: bool | None = None,
    write_outputs: bool = True,
) -> PolicyCheckResult:
    run_dir_path = Path(run_dir).resolve()
    results_path = run_dir_path / "cosmos_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"missing required file: {results_path}")
    results_sha = _sha256_file(results_path)

    raw_results = _read_json(results_path)
    if not isinstance(raw_results, list):
        raise ValueError("cosmos_results.json must be a list")

    review_log_path = (
        Path(review_log).expanduser().resolve()
        if review_log is not None and str(review_log).strip() != ""
        else (run_dir_path / DEFAULT_REVIEW_LOG_NAME)
    )
    review_rows = _load_jsonl(review_log_path)

    latest_review_by_rank: dict[int, dict[str, Any]] = {}
    for row in review_rows:
        try:
            rank = int(row.get("candidate", {}).get("candidate_rank"))
        except Exception:
            continue
        latest_review_by_rank[rank] = row

    effective_policy, source = resolve_policy(
        policy_path=policy_path,
        required_review_severities=required_review_severities,
        require_parse_failure_review=require_parse_failure_review,
        require_error_review=require_error_review,
        allow_stale=allow_stale,
    )
    allow_stale_effective = bool(effective_policy["allow_stale"])
    policy_severities = {str(s).upper() for s in effective_policy["required_review_severities"]}
    require_parse_failure_review_effective = bool(effective_policy["require_parse_failure_review"])
    require_error_review_effective = bool(effective_policy["require_error_review"])

    report_path_resolved = (
        Path(report_path).expanduser().resolve()
        if report_path is not None and str(report_path).strip() != ""
        else (run_dir_path / DEFAULT_POLICY_REPORT_NAME)
    )
    queue_path_resolved = (
        Path(queue_path).expanduser().resolve()
        if queue_path is not None and str(queue_path).strip() != ""
        else (run_dir_path / DEFAULT_REVIEW_QUEUE_NAME)
    )
    snapshot_path_resolved = (
        Path(snapshot_path).expanduser().resolve()
        if snapshot_path is not None and str(snapshot_path).strip() != ""
        else (run_dir_path / DEFAULT_POLICY_SNAPSHOT_NAME)
    )

    generated_at = _utc_now_iso()
    required_review_count = 0
    reviewed_count_valid = 0
    reviewed_count_stale = 0
    missing_review_count = 0
    violations: list[dict[str, Any]] = []
    queue_items: list[dict[str, Any]] = []

    for row in raw_results:
        if not isinstance(row, dict):
            continue
        rank = int(row.get("candidate_rank", -1))
        severity = _normalize_severity(row.get("severity", "NONE"))
        parse_success = bool(row.get("parse_success", False))
        error_text = str(row.get("error", "")).strip()

        required_reasons: list[str] = []
        if severity in policy_severities:
            required_reasons.append("severity_requires_review")
        if require_parse_failure_review_effective and (not parse_success):
            required_reasons.append("parse_failure_requires_review")
        if require_error_review_effective and error_text != "":
            required_reasons.append("error_requires_review")

        if not required_reasons:
            continue

        required_review_count += 1
        review_record = latest_review_by_rank.get(rank)
        has_review = review_record is not None
        stale = False
        valid_review = False
        if review_record is not None:
            recorded_sha = str(review_record.get("run", {}).get("results_sha256", "")).strip()
            stale = bool(recorded_sha != "" and recorded_sha != results_sha)
            if stale:
                reviewed_count_stale += 1
                valid_review = bool(allow_stale_effective)
            else:
                valid_review = True

        if valid_review:
            reviewed_count_valid += 1
            continue

        missing_review_count += 1
        violation_reasons = list(required_reasons)
        if has_review and stale and not allow_stale_effective:
            violation_reasons.append("stale_review")
        if not has_review:
            violation_reasons.append("missing_review")

        item = {
            "candidate_rank": rank,
            "clip_path": str(row.get("clip_path", "")),
            "severity": severity,
            "parse_success": parse_success,
            "error": error_text,
            "required_reasons": required_reasons,
            "violation_reasons": violation_reasons,
            "has_review_record": has_review,
            "stale_review_record": bool(has_review and stale),
        }
        violations.append(item)
        queue_items.append(item)

    queue_items.sort(
        key=lambda item: (
            SEVERITY_ORDER.get(str(item.get("severity", "NONE")).upper(), 9),
            int(item.get("candidate_rank", 10**9)),
        )
    )
    passed = missing_review_count == 0

    summary = {
        "total_candidates": len([r for r in raw_results if isinstance(r, dict)]),
        "required_review_count": required_review_count,
        "reviewed_count_valid": reviewed_count_valid,
        "reviewed_count_stale": reviewed_count_stale,
        "missing_review_count": missing_review_count,
        "passed": passed,
    }
    policy_payload = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "run_dir": str(run_dir_path),
        "results_path": str(results_path),
        "results_sha256": results_sha,
        "review_log_path": str(review_log_path),
        "allow_stale": bool(allow_stale_effective),
        "policy": dict(effective_policy),
        "policy_source": source,
        "summary": summary,
        "violations": violations,
    }
    queue_payload = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "run_dir": str(run_dir_path),
        "results_path": str(results_path),
        "results_sha256": results_sha,
        "review_log_path": str(review_log_path),
        "summary": {
            "queue_size": len(queue_items),
            "required_review_count": required_review_count,
        },
        "items": queue_items,
    }

    if write_outputs:
        _write_json(report_path_resolved, policy_payload)
        _write_json(queue_path_resolved, queue_payload)
        _write_json(
            snapshot_path_resolved,
            {
                "schema_version": POLICY_SCHEMA_VERSION,
                "generated_at_utc": generated_at,
                "run_dir": str(run_dir_path),
                "policy": dict(effective_policy),
                "policy_source": source,
            },
        )
        log.info(
            "Policy check report written: passed=%s required=%d missing=%d",
            passed,
            required_review_count,
            missing_review_count,
        )

    return PolicyCheckResult(
        run_dir=run_dir_path,
        results_path=results_path,
        review_log_path=review_log_path,
        report_path=report_path_resolved,
        queue_path=queue_path_resolved,
        snapshot_path=snapshot_path_resolved,
        policy_source=source,
        passed=passed,
        total_candidates=summary["total_candidates"],
        required_review_count=required_review_count,
        reviewed_count_valid=reviewed_count_valid,
        reviewed_count_stale=reviewed_count_stale,
        missing_review_count=missing_review_count,
        violations=violations,
    )
