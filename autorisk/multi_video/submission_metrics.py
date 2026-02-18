"""Submission metrics aggregation across multiple configured runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from autorisk.dashboard.comparison_utils import positive_min_ttc_values, severity_counts
from autorisk.utils.video_sources import list_video_sources

try:
    from scipy.stats import spearmanr  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    spearmanr = None  # type: ignore[assignment]

SEVERITY_TO_INT = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _as_float(value: Any, default: Any = 0.0) -> Any:
    try:
        return float(value)
    except Exception:
        return default


def _clip_basename(path_text: str) -> str:
    return Path(str(path_text).replace("\\", "/")).name


def ttc_correlation(
    cosmos_results: list[dict[str, Any]],
    ttc_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute Spearman correlation between min TTC and severity rank."""
    if spearmanr is None:
        return {"rho": None, "p_value": None, "n": 0}

    severity_by_clip: dict[str, int] = {}
    for row in cosmos_results:
        if not isinstance(row, dict):
            continue
        clip_name = _clip_basename(str(row.get("clip_path", "")))
        severity = str(row.get("severity", "NONE")).upper()
        if clip_name == "" or severity not in SEVERITY_TO_INT:
            continue
        severity_by_clip[clip_name] = SEVERITY_TO_INT[severity]

    xs: list[float] = []
    ys: list[int] = []
    for row in ttc_results:
        if not isinstance(row, dict):
            continue
        clip_name = _clip_basename(str(row.get("clip_path", "")))
        if clip_name not in severity_by_clip:
            continue
        min_ttc = _as_float(row.get("min_ttc", -1.0), default=-1.0)
        if min_ttc <= 0:
            continue
        xs.append(min_ttc)
        ys.append(severity_by_clip[clip_name])

    if len(xs) < 3:
        return {"rho": None, "p_value": None, "n": len(xs)}

    stat = spearmanr(xs, ys)
    return {
        "rho": _as_float(getattr(stat, "statistic", None), default=None),
        "p_value": _as_float(getattr(stat, "pvalue", None), default=None),
        "n": len(xs),
    }


def source_submission_summary(*, repo_root: str | Path, source: dict[str, Any]) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output_dir = root / str(source["output_dir"])
    cosmos_results = _read_json(output_dir / "cosmos_results.json", [])
    eval_report = _read_json(output_dir / "eval_report.json", {})
    grounding_report = _read_json(output_dir / "grounding_report.json", {})
    ttc_results = _read_json(output_dir / "ttc_results.json", [])
    analysis_report = _read_json(output_dir / "analysis_report.json", {})

    cosmos = cosmos_results if isinstance(cosmos_results, list) else []
    ttc = ttc_results if isinstance(ttc_results, list) else []
    eval_dict = eval_report if isinstance(eval_report, dict) else {}
    grounding_dict = grounding_report if isinstance(grounding_report, dict) else {}
    analysis_dict = analysis_report if isinstance(analysis_report, dict) else {}

    counts = severity_counts(cosmos)
    clip_count = len(cosmos)
    parse_ok = sum(
        1 for row in cosmos if isinstance(row, dict) and bool(row.get("parse_success", False))
    )
    parse_rate = (parse_ok / clip_count) if clip_count else 0.0
    confidences = [_as_float(row.get("confidence", 0.0)) for row in cosmos if isinstance(row, dict)]
    ttc_values = positive_min_ttc_values(ttc)
    ttc_corr = ttc_correlation(cosmos, ttc)

    fused_signal = {}
    for item in analysis_dict.get("signal_analysis", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("signal_name", "")).lower() == "fused":
            fused_signal = {
                "spearman_rho": _as_float(item.get("spearman_rho", 0.0)),
                "spearman_p": _as_float(item.get("spearman_p", 0.0)),
            }
            break

    return {
        "name": str(source["name"]),
        "config_path": str(source["config_path"]),
        "output_dir": str(output_dir),
        "available": bool((output_dir / "cosmos_results.json").exists()),
        "clip_count": clip_count,
        "parse_success_count": parse_ok,
        "parse_success_rate": round(parse_rate, 6),
        "severity_counts": counts,
        "accuracy": _as_float(eval_dict.get("accuracy", 0.0)),
        "macro_f1": _as_float(eval_dict.get("macro_f1", 0.0)),
        "checklist_mean_total": _as_float(
            (eval_dict.get("checklist_means", {}) or {}).get("mean_total", 0.0)
        ),
        "mean_confidence": round(mean(confidences), 6) if confidences else 0.0,
        "grounding_mean_score": _as_float(grounding_dict.get("mean_grounding_score", 0.0)),
        "ttc": {
            "n_positive_min_ttc": len(ttc_values),
            "mean_min_ttc": round(mean(ttc_values), 6) if ttc_values else None,
            "min_min_ttc": min(ttc_values) if ttc_values else None,
            "spearman_rho_vs_severity": ttc_corr["rho"],
            "spearman_p_value": ttc_corr["p_value"],
            "spearman_n": ttc_corr["n"],
        },
        "fused_signal": fused_signal,
    }


def build_submission_metrics(
    *,
    repo_root: str | Path,
    only: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    sources = list_video_sources(repo_root=root, only=only)
    summaries = [source_submission_summary(repo_root=root, source=source) for source in sources]

    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now_iso(),
        "sources_total": len(summaries),
        "sources_available": sum(1 for summary in summaries if bool(summary["available"])),
        "clips_total": sum(int(summary["clip_count"]) for summary in summaries),
        "sources": summaries,
    }


def write_submission_metrics(payload: dict[str, Any], *, output_path: str | Path) -> Path:
    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path

