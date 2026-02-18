"""Build auditable evidence bundles from a completed AutoRisk-RM run."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from autorisk import __version__ as _PACKAGE_VERSION
from autorisk.cosmos.infer import _parse_markdown_response, _repair_truncated_json
from autorisk.cosmos.prompt import (
    SUPPLEMENT_SYSTEM_PROMPT,
    SUPPLEMENT_USER_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)

SCHEMA_VERSION = 1
_CHECKSUMS_FILENAME = "checksums.sha256.txt"
_MANIFEST_FILENAME = "manifest.json"
_TRACE_FILENAME = "decision_trace.jsonl"
_FINALIZE_RECORD_FILENAME = "finalize_record.json"
_VALIDATE_REPORT_FILENAME = "audit_validate_report.json"
SCHEMA_COMPATIBILITY = (
    "Backward-compatible additive changes only within the same schema_version. "
    "Consumers must ignore unknown fields. Breaking changes increment schema_version."
)


@dataclass
class ParseAudit:
    parse_replay_success: bool
    parse_stage: str
    repair_applied: bool
    json_repair_log: list[dict[str, Any]]


@dataclass
class AuditPackResult:
    output_dir: Path
    zip_path: Path | None
    manifest_path: Path
    decision_trace_path: Path
    checksums_path: Path
    checksums_sha256: str
    records: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return default


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(raw)
    except Exception:
        return default


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _package_version() -> str:
    try:
        return str(importlib.metadata.version("autorisk-rm"))
    except Exception:
        return _PACKAGE_VERSION


def _environment_info() -> dict[str, Any]:
    pkgs: dict[str, str] = {}
    for name in [
        "autorisk-rm",
        "omegaconf",
        "numpy",
        "opencv-python",
        "ultralytics",
        "torch",
        "transformers",
    ]:
        try:
            pkgs[name] = str(importlib.metadata.version(name))
        except Exception:
            continue
    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": pkgs,
    }


def _find_git_root(start: Path) -> Path | None:
    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return p
    return None


def _git_info(start: Path) -> dict[str, Any]:
    root = _find_git_root(start.resolve())
    if root is None:
        return {"is_git_repo": False}

    def _run(args: list[str]) -> str:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                return ""
            return str((proc.stdout or "").strip())
        except Exception:
            return ""

    commit = _run(["rev-parse", "HEAD"])
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    describe = _run(["describe", "--tags", "--always", "--dirty"])
    status = _run(["status", "--porcelain"])
    return {
        "is_git_repo": True,
        "git_root": str(root),
        "commit": commit,
        "branch": branch,
        "describe": describe,
        "dirty": bool(status.strip()),
    }


def _operator_info() -> dict[str, str]:
    user = (
        os.environ.get("AUTORISK_OPERATOR")
        or os.environ.get("USERNAME")
        or os.environ.get("USER")
        or "unknown"
    )
    host = os.environ.get("COMPUTERNAME") or socket.gethostname() or "unknown"
    return {"user": str(user), "host": str(host)}


def _prompt_info() -> dict[str, Any]:
    primary_combined = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"
    supplement_combined = f"{SUPPLEMENT_SYSTEM_PROMPT}\n\n{SUPPLEMENT_USER_PROMPT}"
    return {
        "primary": {
            "system_sha256": _sha256_text(SYSTEM_PROMPT),
            "user_sha256": _sha256_text(USER_PROMPT),
            "combined_sha256": _sha256_text(primary_combined),
        },
        "supplement": {
            "system_sha256": _sha256_text(SUPPLEMENT_SYSTEM_PROMPT),
            "user_sha256": _sha256_text(SUPPLEMENT_USER_PROMPT),
            "combined_sha256": _sha256_text(supplement_combined),
        },
    }


def _model_info(cfg: Any) -> dict[str, Any]:
    backend = str(cfg.cosmos.get("backend", "unknown")) if cfg is not None else "unknown"
    model_name = "unknown"
    model_params: dict[str, Any] = {}
    if backend == "local" and cfg is not None:
        local_cfg = cfg.cosmos.get("local", {})
        model_name = str(local_cfg.get("model_name", "unknown"))
        model_params = {
            "max_new_tokens": _safe_int(local_cfg.get("max_new_tokens", 0)),
            "temperature": _safe_float(local_cfg.get("temperature", 0.0)),
            "torch_dtype": str(local_cfg.get("torch_dtype", "unknown")),
            "local_fps": _safe_int(cfg.cosmos.get("local_fps", 0)),
        }
    elif backend == "api" and cfg is not None:
        api_cfg = cfg.cosmos.get("api", {})
        model_name = str(api_cfg.get("model", "unknown"))
        model_params = {
            "base_url": str(api_cfg.get("base_url", "")),
            "max_tokens": _safe_int(api_cfg.get("max_tokens", 0)),
            "temperature": _safe_float(api_cfg.get("temperature", 0.0)),
            "top_p": _safe_float(api_cfg.get("top_p", 0.0)),
        }
    return {
        "backend": backend,
        "model_name": model_name,
        "params": model_params,
    }


def _path_keys(path_like: str) -> set[str]:
    text = str(path_like or "").strip()
    if text == "":
        return set()
    p = Path(text)
    out: set[str] = {
        text.replace("\\", "/").lower(),
        p.name.replace("\\", "/").lower(),
    }
    try:
        resolved = str(p.resolve())
        out.add(resolved.replace("\\", "/").lower())
    except Exception:
        pass
    return out


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
    for candidate in candidates:
        if candidate.exists():
            try:
                return candidate.resolve()
            except Exception:
                return candidate
    return p


def _load_candidate_index(candidates_csv: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with candidates_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_path = str(row.get("clip_path", "")).strip()
            parsed_scores: dict[str, Any] = {}
            raw_scores = row.get("signal_scores", "")
            if isinstance(raw_scores, str) and raw_scores.strip() != "":
                try:
                    literal = ast.literal_eval(raw_scores)
                    if isinstance(literal, dict):
                        parsed_scores = dict(literal)
                except Exception:
                    parsed_scores = {}
            entry = {
                "rank": _safe_int(row.get("rank", 0)),
                "peak_time_sec": _safe_float(row.get("peak_time_sec", 0.0)),
                "start_sec": _safe_float(row.get("start_sec", 0.0)),
                "end_sec": _safe_float(row.get("end_sec", 0.0)),
                "fused_score": _safe_float(row.get("fused_score", 0.0)),
                "audio": _safe_float(parsed_scores.get("audio", 0.0)),
                "motion": _safe_float(parsed_scores.get("motion", 0.0)),
                "proximity": _safe_float(parsed_scores.get("proximity", 0.0)),
            }
            for key in _path_keys(clip_path):
                out.setdefault(key, entry)
    return out


def _parse_json_stage(text: str) -> ParseAudit:
    log_rows: list[dict[str, Any]] = []

    def _record(step: str, success: bool, detail: str = "") -> None:
        log_rows.append({"step": step, "success": bool(success), "detail": str(detail)})

    def _try_json(step: str, payload: str) -> bool:
        try:
            json.loads(payload)
            _record(step, True)
            return True
        except Exception as exc:
            _record(step, False, str(exc)[:200])
            return False

    raw = str(text or "")
    if raw.strip() == "":
        _record("empty_response", False, "raw answer is empty")
        return ParseAudit(
            parse_replay_success=False,
            parse_stage="empty_response",
            repair_applied=False,
            json_repair_log=log_rows,
        )

    if _try_json("direct_json", raw):
        return ParseAudit(True, "direct_json", False, log_rows)

    stripped = raw.strip()
    if stripped.startswith("json"):
        trimmed = stripped[4:].strip()
        if _try_json("json_prefix", trimmed):
            return ParseAudit(True, "json_prefix", False, log_rows)

    fence_match = re.search(r"```(?:json)?\s*\n?(.*)\n?```", raw, re.DOTALL)
    if fence_match and _try_json("markdown_fence", fence_match.group(1).strip()):
        return ParseAudit(True, "markdown_fence", False, log_rows)

    open_fence = re.search(r"```(?:json)?\s*\n?(.*)", raw, re.DOTALL)
    if open_fence:
        inner = open_fence.group(1).strip()
        if _try_json("open_fence", inner):
            return ParseAudit(True, "open_fence", False, log_rows)
        repaired = _repair_truncated_json(inner)
        if _try_json("open_fence_repair", repaired):
            return ParseAudit(True, "open_fence_repair", True, log_rows)

    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match and _try_json("brace_extract", brace_match.group(0)):
        return ParseAudit(True, "brace_extract", False, log_rows)

    brace_start = re.search(r"\{.*", raw, re.DOTALL)
    if brace_start:
        repaired = _repair_truncated_json(brace_start.group(0))
        if _try_json("brace_repair", repaired):
            return ParseAudit(True, "brace_repair", True, log_rows)

    md_data = _parse_markdown_response(raw)
    if md_data is not None:
        _record("markdown_fallback", True, "parsed with markdown fallback")
        return ParseAudit(True, "markdown_fallback", False, log_rows)

    _record("parse_failed", False, "all parser stages failed")
    return ParseAudit(False, "parse_failed", False, log_rows)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _sanitize_finalize_record_for_pack(payload: dict[str, Any]) -> dict[str, Any]:
    """Sanitize finalize_record for PACK-internal contract (no handoff_* hashes)."""
    sanitized = dict(payload)
    for key in [
        "handoff_path",
        "handoff_checksums_sha256",
        "handoff_pack_zip_sha256",
        "handoff_verifier_bundle_zip_sha256",
    ]:
        sanitized[key] = ""

    anchor_checksums = str(sanitized.get("handoff_anchor_checksums_sha256", "")).strip()
    anchor_bundle = str(sanitized.get("handoff_anchor_verifier_bundle_zip_sha256", "")).strip()
    if (anchor_checksums == "") != (anchor_bundle == ""):
        sanitized["handoff_anchor_checksums_sha256"] = ""
        sanitized["handoff_anchor_verifier_bundle_zip_sha256"] = ""
    return sanitized


def _copy_finalize_record_for_pack(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    try:
        loaded = _read_json(src)
    except Exception as exc:
        log.warning("Skipping finalize_record copy (invalid JSON): %s (%s)", src, str(exc)[:200])
        return False
    if not isinstance(loaded, dict):
        log.warning("Skipping finalize_record copy (must be JSON object): %s", src)
        return False
    _write_json(dst, _sanitize_finalize_record_for_pack(loaded))
    return True


def _collect_payload_file_entries(
    root: Path,
    *,
    exclude_names: set[str] | None = None,
    exclude_paths: set[str] | None = None,
) -> list[dict[str, Any]]:
    exclude = set(exclude_names or set())
    exclude_rel = {str(p).replace("\\", "/") for p in (exclude_paths or set())}
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if path.name in exclude:
            continue
        if rel in exclude_rel:
            continue
        rows.append(
            {
                "path": rel,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _write_checksums(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [f"{row['sha256']}  {row['path']}" for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _build_severity_histogram(results: list[dict[str, Any]]) -> dict[str, int]:
    hist = {"NONE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for row in results:
        sev = str(row.get("severity", "NONE")).upper()
        hist[sev] = hist.get(sev, 0) + 1
    return hist


def build_audit_pack(
    *,
    run_dir: str | Path,
    cfg: Any,
    output_dir: str | Path | None = None,
    input_video: str | Path | None = None,
    review_log: str | Path | None = None,
    include_clips: bool = True,
    create_zip: bool = True,
) -> AuditPackResult:
    """Build an auditable evidence pack from a completed run directory."""
    run_dir_path = Path(run_dir).resolve()
    if not run_dir_path.exists():
        raise FileNotFoundError(f"run directory not found: {run_dir_path}")

    cosmos_results_path = run_dir_path / "cosmos_results.json"
    candidates_csv_path = run_dir_path / "candidates.csv"
    if not cosmos_results_path.exists():
        raise FileNotFoundError(f"missing required file: {cosmos_results_path}")
    if not candidates_csv_path.exists():
        raise FileNotFoundError(f"missing required file: {candidates_csv_path}")

    ts_slug = _utc_now_slug()
    if output_dir is None:
        output_dir_path = run_dir_path / f"audit_pack_{ts_slug}"
    else:
        output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    run_artifacts_dir = output_dir_path / "run_artifacts"
    run_artifacts_dir.mkdir(parents=True, exist_ok=True)

    cosmos_results = _read_json(cosmos_results_path)
    if not isinstance(cosmos_results, list):
        raise ValueError("cosmos_results.json must be a list")
    candidate_index = _load_candidate_index(candidates_csv_path)

    model_info = _model_info(cfg)
    prompt_info = _prompt_info()
    operator = _operator_info()
    generated_at = _utc_now_iso()
    environment = _environment_info()
    source = _git_info(run_dir_path)

    input_video_path_str = ""
    input_video_sha256 = ""
    if input_video is not None and str(input_video).strip() != "":
        p = Path(input_video).expanduser()
        if p.exists():
            input_video_path_str = str(p.resolve())
            input_video_sha256 = _sha256_file(p)
        else:
            input_video_path_str = str(p)

    clip_hash_cache: dict[str, str] = {}
    copied_clips: set[str] = set()
    traces: list[dict[str, Any]] = []
    for i, row in enumerate(cosmos_results, start=1):
        if not isinstance(row, dict):
            continue
        clip_raw = str(row.get("clip_path", ""))
        clip_abs = _resolve_clip_path(clip_raw, run_dir=run_dir_path)

        candidate_meta: dict[str, Any] = {}
        lookup_keys = set()
        lookup_keys.update(_path_keys(clip_raw))
        lookup_keys.update(_path_keys(str(clip_abs)))
        for key in lookup_keys:
            if key in candidate_index:
                candidate_meta = candidate_index[key]
                break

        clip_sha = ""
        if clip_abs.exists() and clip_abs.is_file():
            cache_key = str(clip_abs)
            clip_sha = clip_hash_cache.get(cache_key, "")
            if clip_sha == "":
                clip_sha = _sha256_file(clip_abs)
                clip_hash_cache[cache_key] = clip_sha
            if include_clips:
                clip_rel = f"clips/{clip_abs.name}"
                if clip_rel not in copied_clips:
                    copied_clips.add(clip_rel)
                    _copy_if_exists(clip_abs, output_dir_path / clip_rel)

        raw_answer = str(row.get("raw_answer", ""))
        parse_audit = _parse_json_stage(raw_answer)

        rank = _safe_int(row.get("candidate_rank", candidate_meta.get("rank", i)))
        trace = {
            "schema_version": SCHEMA_VERSION,
            "trace_id": f"candidate-{rank:03d}",
            "candidate_rank": rank,
            "clip_path": clip_raw,
            "clip_sha256": clip_sha,
            "input_video_sha256": input_video_sha256,
            "event_time_sec": {
                "peak": _safe_float(row.get("peak_time_sec", candidate_meta.get("peak_time_sec", 0.0))),
                "start": _safe_float(candidate_meta.get("start_sec", 0.0)),
                "end": _safe_float(candidate_meta.get("end_sec", 0.0)),
            },
            "candidate_scores": {
                "fused": _safe_float(row.get("fused_score", candidate_meta.get("fused_score", 0.0))),
                "audio": _safe_float(candidate_meta.get("audio", 0.0)),
                "motion": _safe_float(candidate_meta.get("motion", 0.0)),
                "proximity": _safe_float(candidate_meta.get("proximity", 0.0)),
            },
            "model": model_info,
            "prompt": prompt_info,
            "parsing": {
                "parse_success_recorded": bool(row.get("parse_success", False)),
                "parse_replay_success": parse_audit.parse_replay_success,
                "parse_stage": parse_audit.parse_stage,
                "repair_applied": parse_audit.repair_applied,
                "json_repair_log": parse_audit.json_repair_log,
            },
            "final_assessment": {
                "severity": str(row.get("severity", "NONE")).upper(),
                "confidence": _safe_float(row.get("confidence", 0.0)),
                "hazards": row.get("hazards", []),
                "causal_reasoning": str(row.get("causal_reasoning", "")),
                "short_term_prediction": str(row.get("short_term_prediction", "")),
                "recommended_action": str(row.get("recommended_action", "")),
                "evidence": row.get("evidence", []),
                "error": str(row.get("error", "")),
            },
            "raw_response": raw_answer,
            "audit_context": {
                "generated_at_utc": generated_at,
                "generated_by": operator,
            },
        }
        traces.append(trace)

    trace_path = output_dir_path / _TRACE_FILENAME
    _write_jsonl(trace_path, traces)

    copied_files: list[str] = []
    run_files = [
        cosmos_results_path,
        candidates_csv_path,
        run_dir_path / "run_summary.json",
        run_dir_path / "submission_metrics.json",
        run_dir_path / _FINALIZE_RECORD_FILENAME,
        run_dir_path / _VALIDATE_REPORT_FILENAME,
        run_dir_path / "cosmos_results_reviewed.json",
        run_dir_path / "review_apply_report.json",
        run_dir_path / "review_diff_report.json",
        run_dir_path / "policy_report.json",
        run_dir_path / "review_queue.json",
        run_dir_path / "policy_snapshot.json",
        run_dir_path / "eval_report.json",
        run_dir_path / "ablation_results.json",
        run_dir_path / "analysis_report.json",
        run_dir_path / "report.html",
        run_dir_path / "report.md",
    ]
    for src in run_files:
        dst = run_artifacts_dir / src.name
        copied = False
        if src.name == _FINALIZE_RECORD_FILENAME:
            copied = _copy_finalize_record_for_pack(src, dst)
        else:
            copied = _copy_if_exists(src, dst)
        if copied:
            copied_files.append(dst.relative_to(output_dir_path).as_posix())

    human_review_files: list[str] = []
    review_candidates: list[Path] = []
    if review_log is not None and str(review_log).strip() != "":
        review_candidates.append(Path(review_log).expanduser())
    else:
        review_candidates.extend(
            [
                run_dir_path / "review_log.jsonl",
                run_dir_path / "human_review.jsonl",
                run_dir_path / "approval_log.jsonl",
            ]
        )
    for candidate in review_candidates:
        if candidate.exists() and candidate.is_file():
            dst = output_dir_path / "human_review" / candidate.name
            if _copy_if_exists(candidate, dst):
                human_review_files.append(dst.relative_to(output_dir_path).as_posix())
            break

    review_diff_summary = {
        "applied": 0,
        "override_count": 0,
        "no_change_count": 0,
    }
    review_diff_path = run_dir_path / "review_diff_report.json"
    if review_diff_path.exists() and review_diff_path.is_file():
        try:
            review_diff_obj = _read_json(review_diff_path)
            if isinstance(review_diff_obj, dict):
                review_diff_summary = {
                    "applied": _safe_int(review_diff_obj.get("applied", 0)),
                    "override_count": _safe_int(review_diff_obj.get("override_count", 0)),
                    "no_change_count": _safe_int(review_diff_obj.get("no_change_count", 0)),
                }
        except Exception:
            review_diff_summary = {
                "applied": 0,
                "override_count": 0,
                "no_change_count": 0,
            }

    cfg_snapshot = {}
    if cfg is not None:
        try:
            cfg_snapshot = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            cfg_snapshot = {}
    _write_json(output_dir_path / "config_snapshot.json", cfg_snapshot)

    payload_entries = _collect_payload_file_entries(
        output_dir_path,
        exclude_names={_MANIFEST_FILENAME, _CHECKSUMS_FILENAME},
        exclude_paths={
            f"run_artifacts/{_FINALIZE_RECORD_FILENAME}",
            f"run_artifacts/{_VALIDATE_REPORT_FILENAME}",
        },
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "schema": {
            "version": SCHEMA_VERSION,
            "compatibility": SCHEMA_COMPATIBILITY,
        },
        "generated_at_utc": generated_at,
        "generated_by": operator,
        "tool": {"name": "autorisk-rm", "version": _package_version()},
        "source": source,
        "environment": environment,
        "run": {
            "run_dir": str(run_dir_path),
            "run_name": run_dir_path.name,
            "input_video_path": input_video_path_str,
            "input_video_sha256": input_video_sha256,
            "results_file": str(cosmos_results_path),
            "candidates_file": str(candidates_csv_path),
        },
        "model": model_info,
        "prompt": prompt_info,
        "summary": {
            "records": len(traces),
            "parse_success_recorded": sum(1 for r in traces if r["parsing"]["parse_success_recorded"]),
            "parse_replay_success": sum(1 for r in traces if r["parsing"]["parse_replay_success"]),
            "severity_histogram": _build_severity_histogram(cosmos_results),
            "clips_included": len(copied_clips),
            "run_files_included": copied_files,
            "human_review_files_included": human_review_files,
            "review_diff": review_diff_summary,
        },
        "payload_files": payload_entries,
    }
    manifest_path = output_dir_path / _MANIFEST_FILENAME
    _write_json(manifest_path, manifest)

    checksum_entries = _collect_payload_file_entries(
        output_dir_path,
        exclude_names={_CHECKSUMS_FILENAME},
        exclude_paths={
            f"run_artifacts/{_FINALIZE_RECORD_FILENAME}",
            f"run_artifacts/{_VALIDATE_REPORT_FILENAME}",
        },
    )
    checksums_path = output_dir_path / _CHECKSUMS_FILENAME
    _write_checksums(checksums_path, checksum_entries)
    checksums_sha256 = _sha256_file(checksums_path)

    zip_path: Path | None = None
    if create_zip:
        zip_path = output_dir_path.parent / f"{output_dir_path.name}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(output_dir_path.rglob("*")):
                if path.is_file():
                    arcname = path.relative_to(output_dir_path).as_posix()
                    zf.write(path, arcname=arcname)
        log.info("Wrote audit pack zip: %s", zip_path)

    log.info(
        "Audit pack created: dir=%s, records=%d, clips=%d",
        output_dir_path,
        len(traces),
        len(copied_clips),
    )
    return AuditPackResult(
        output_dir=output_dir_path,
        zip_path=zip_path,
        manifest_path=manifest_path,
        decision_trace_path=trace_path,
        checksums_path=checksums_path,
        checksums_sha256=checksums_sha256,
        records=len(traces),
    )
