from __future__ import annotations

import hashlib
import json
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)

REVIEW_SCHEMA_VERSION = 1
DEFAULT_REVIEW_LOG_NAME = "review_log.jsonl"
DEFAULT_REVIEWED_RESULTS_NAME = "cosmos_results_reviewed.json"
DEFAULT_REVIEW_APPLY_REPORT_NAME = "review_apply_report.json"
DEFAULT_REVIEW_DIFF_REPORT_NAME = "review_diff_report.json"

SEVERITIES = {"NONE", "LOW", "MEDIUM", "HIGH"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")


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


def _normalize_severity(raw: str) -> str:
    s = str(raw or "").strip().upper()
    if s not in SEVERITIES:
        raise ValueError(f"Invalid severity: {raw} (allowed: {sorted(SEVERITIES)})")
    return s


def _operator() -> dict[str, str]:
    user = (
        os.environ.get("AUTORISK_OPERATOR")
        or os.environ.get("USERNAME")
        or os.environ.get("USER")
        or "unknown"
    )
    host = os.environ.get("COMPUTERNAME") or socket.gethostname() or "unknown"
    return {"user": str(user), "host": str(host)}


def _resolve_clip_path(raw_clip_path: str, *, run_dir: Path) -> Path:
    text = str(raw_clip_path or "").strip()
    if text == "":
        return Path("")
    p = Path(text)
    candidates = [p]
    if not p.is_absolute():
        candidates.append(Path.cwd() / p)
        candidates.append(run_dir / p)
        candidates.append(run_dir / "clips" / p.name)
    for c in candidates:
        if c.exists():
            try:
                return c.resolve()
            except Exception:
                return c
    return p


def _find_candidate(results: list[dict[str, Any]], rank: int) -> dict[str, Any]:
    available: set[int] = set()
    for row in results:
        if not isinstance(row, dict):
            continue
        if "candidate_rank" not in row:
            continue
        try:
            current_rank = int(row.get("candidate_rank"))
        except Exception:
            continue
        available.add(current_rank)
        if current_rank == int(rank):
            return row
    available_sorted = sorted(available)
    raise KeyError(
        f"candidate_rank={rank} not found in cosmos_results.json "
        f"(available: {available_sorted[:50]})"
    )


def load_review_log(*, run_dir: str | Path, log_path: str | Path | None = None) -> list[dict[str, Any]]:
    run_dir_path = Path(run_dir).resolve()
    path = Path(log_path).expanduser().resolve() if log_path else (run_dir_path / DEFAULT_REVIEW_LOG_NAME)
    return _load_jsonl(path)


def append_review_decision(
    *,
    run_dir: str | Path,
    candidate_rank: int,
    severity_after: str,
    reason: str,
    evidence_refs: list[str] | None = None,
    operator_user: str | None = None,
    log_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any], str]:
    """Append one review decision line to review_log.jsonl."""
    run_dir_path = Path(run_dir).resolve()
    results_path = run_dir_path / "cosmos_results.json"
    candidates_path = run_dir_path / "candidates.csv"

    if not results_path.exists():
        raise FileNotFoundError(f"missing required file: {results_path}")

    results = _read_json(results_path)
    if not isinstance(results, list):
        raise ValueError("cosmos_results.json must be a list")

    row = _find_candidate(results, int(candidate_rank))

    op = _operator()
    if operator_user and str(operator_user).strip():
        op["user"] = str(operator_user).strip()

    severity_before = _normalize_severity(str(row.get("severity", "NONE")))
    severity_after_norm = _normalize_severity(severity_after)

    clip_raw = str(row.get("clip_path", ""))
    clip_abs = _resolve_clip_path(clip_raw, run_dir=run_dir_path)

    clip_sha256 = ""
    if clip_abs.exists() and clip_abs.is_file():
        try:
            clip_sha256 = _sha256_file(clip_abs)
        except Exception:
            clip_sha256 = ""

    record: dict[str, Any] = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "timestamp_utc": _utc_now_iso(),
        "operator": op,
        "run": {
            "run_dir": str(run_dir_path),
            "results_sha256": _sha256_file(results_path),
            "candidates_sha256": _sha256_file(candidates_path) if candidates_path.exists() else "",
        },
        "candidate": {
            "candidate_rank": int(candidate_rank),
            "clip_path": clip_raw,
            "clip_sha256": clip_sha256,
        },
        "decision_before": {
            "severity": severity_before,
            "confidence": row.get("confidence", None),
        },
        "decision_after": {
            "severity": severity_after_norm,
        },
        "reason": str(reason or "").strip(),
        "evidence_refs": list(evidence_refs or []),
    }

    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    record_sha256 = _sha256_text(canonical)

    path = Path(log_path).expanduser().resolve() if log_path else (run_dir_path / DEFAULT_REVIEW_LOG_NAME)
    _append_jsonl(path, record)
    log.info(
        "Appended review decision: rank=%s severity=%s sha=%s",
        candidate_rank,
        severity_after_norm,
        record_sha256,
    )
    return path, record, record_sha256


@dataclass
class ReviewApplyResult:
    input_results: Path
    output_results: Path
    log_path: Path
    diff_report_path: Path
    applied: int
    skipped_stale: int
    skipped_missing: int


def apply_review_overrides(
    *,
    run_dir: str | Path,
    log_path: str | Path | None = None,
    output_path: str | Path | None = None,
    allow_stale: bool = False,
    write_report: bool = True,
) -> ReviewApplyResult:
    """Apply latest review decision per candidate to a reviewed results file."""
    run_dir_path = Path(run_dir).resolve()
    results_path = run_dir_path / "cosmos_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"missing required file: {results_path}")

    current_results_sha = _sha256_file(results_path)

    log_file = Path(log_path).expanduser().resolve() if log_path else (run_dir_path / DEFAULT_REVIEW_LOG_NAME)
    rows = _load_jsonl(log_file)

    latest: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            rank = int(row.get("candidate", {}).get("candidate_rank"))
        except Exception:
            continue
        latest[rank] = row

    results = _read_json(results_path)
    if not isinstance(results, list):
        raise ValueError("cosmos_results.json must be a list")

    index: dict[int, dict[str, Any]] = {}
    for row in results:
        if isinstance(row, dict) and "candidate_rank" in row:
            try:
                index[int(row["candidate_rank"])] = row
            except Exception:
                continue

    applied = 0
    skipped_stale = 0
    skipped_missing = 0
    applied_at = _utc_now_iso()
    transition_histogram: dict[str, int] = {}
    changed_candidates: list[dict[str, Any]] = []
    unchanged_candidates: list[int] = []

    for rank, rec in latest.items():
        if rank not in index:
            skipped_missing += 1
            continue

        rec_results_sha = str(rec.get("run", {}).get("results_sha256", "")).strip()
        if rec_results_sha and rec_results_sha != current_results_sha:
            if not allow_stale:
                skipped_stale += 1
                continue

        after = rec.get("decision_after", {}) or {}
        new_sev = _normalize_severity(str(after.get("severity", index[rank].get("severity", "NONE"))))

        canonical = json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        rec_sha = _sha256_text(canonical)

        row = index[rank]
        before_sev = _normalize_severity(str(row.get("severity", "NONE")))
        row["severity"] = new_sev
        row["review"] = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "applied_at_utc": applied_at,
            "record_sha256": rec_sha,
            "log_path": str(log_file),
            "timestamp_utc": rec.get("timestamp_utc", ""),
            "operator": rec.get("operator", {}),
            "decision_before": rec.get("decision_before", {}),
            "decision_after": rec.get("decision_after", {}),
            "reason": rec.get("reason", ""),
            "evidence_refs": rec.get("evidence_refs", []),
            "stale": bool(rec_results_sha and rec_results_sha != current_results_sha),
            "severity_before_at_apply": before_sev,
        }
        if before_sev != new_sev:
            transition = f"{before_sev}->{new_sev}"
            transition_histogram[transition] = transition_histogram.get(transition, 0) + 1
            changed_candidates.append(
                {
                    "candidate_rank": rank,
                    "severity_before": before_sev,
                    "severity_after": new_sev,
                    "record_sha256": rec_sha,
                    "reason": rec.get("reason", ""),
                }
            )
        else:
            unchanged_candidates.append(rank)
        applied += 1

    out_path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else (run_dir_path / DEFAULT_REVIEWED_RESULTS_NAME)
    )
    _write_json(out_path, results)
    diff_report_path = run_dir_path / DEFAULT_REVIEW_DIFF_REPORT_NAME

    if write_report:
        report = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "applied_at_utc": applied_at,
            "run_dir": str(run_dir_path),
            "input_results": str(results_path),
            "input_results_sha256": current_results_sha,
            "review_log": str(log_file),
            "applied": applied,
            "skipped_stale": skipped_stale,
            "skipped_missing": skipped_missing,
            "output_results": str(out_path),
        }
        _write_json(run_dir_path / DEFAULT_REVIEW_APPLY_REPORT_NAME, report)
        diff_report = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "applied_at_utc": applied_at,
            "run_dir": str(run_dir_path),
            "input_results": str(results_path),
            "input_results_sha256": current_results_sha,
            "review_log": str(log_file),
            "output_results": str(out_path),
            "applied": applied,
            "override_count": len(changed_candidates),
            "no_change_count": len(unchanged_candidates),
            "transition_histogram": transition_histogram,
            "changed_candidates": changed_candidates,
        }
        _write_json(diff_report_path, diff_report)

    return ReviewApplyResult(
        input_results=results_path,
        output_results=out_path,
        log_path=log_file,
        diff_report_path=diff_report_path,
        applied=applied,
        skipped_stale=skipped_stale,
        skipped_missing=skipped_missing,
    )
