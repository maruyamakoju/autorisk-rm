"""Pure helpers for cross-run comparison dashboards."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

RUN_LABELS: dict[str, str] = {
    "public_run": "UK Urban",
    "japan_run": "Japan",
    "winter_run": "Winter/Snow",
    "us_highway_run": "US Highway",
}
SEVERITY_ORDER: tuple[str, str, str, str] = ("NONE", "LOW", "MEDIUM", "HIGH")
SIGNAL_NAMES: tuple[str, str, str] = ("audio", "motion", "proximity")

# Centralized severity color scheme (used across all dashboard pages)
SEVERITY_COLORS: dict[str, str] = {
    "NONE": "#6B7280",    # gray
    "LOW": "#10B981",     # green
    "MEDIUM": "#F59E0B",  # orange/yellow
    "HIGH": "#EF4444",    # red
}


def discover_runs(
    *,
    outputs_root: Path,
    loader: Callable[[str], dict[str, Any]] | None = None,
    run_labels: dict[str, str] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Discover run directories that have cosmos_results and load them."""
    if loader is None:
        from autorisk.dashboard.data_loader import load_data

        loader_fn = load_data
    else:
        loader_fn = loader

    labels = run_labels or RUN_LABELS
    runs: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    if not outputs_root.exists():
        return runs, skipped

    for entry in sorted(outputs_root.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "cosmos_results.json").exists():
            continue

        label = labels.get(entry.name, entry.name)
        try:
            data = loader_fn(str(entry))
        except Exception as exc:
            skipped.append(f"{label}: {str(exc)[:140]}")
            continue
        runs[label] = data
    return runs, skipped


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def compute_run_kpis(run_data: dict[str, Any]) -> dict[str, float | int]:
    """Extract lightweight KPIs for one run."""
    cosmos = run_data.get("cosmos_results", [])
    cosmos_list = cosmos if isinstance(cosmos, list) else []
    eval_report = run_data.get("eval_report", {})
    eval_dict = eval_report if isinstance(eval_report, dict) else {}
    checklist = eval_dict.get("checklist_means", {})
    checklist_dict = checklist if isinstance(checklist, dict) else {}

    n_clips = len(cosmos_list)
    parse_ok = sum(1 for r in cosmos_list if isinstance(r, dict) and bool(r.get("parse_success", False)))
    return {
        "clips": n_clips,
        "parse_ok": parse_ok,
        "accuracy": _as_float(eval_dict.get("accuracy", 0.0)),
        "macro_f1": _as_float(eval_dict.get("macro_f1", 0.0)),
        "checklist_mean_total": _as_float(checklist_dict.get("mean_total", 0.0)),
    }


def severity_counts(cosmos_results: list[dict[str, Any]]) -> dict[str, int]:
    counts = {severity: 0 for severity in SEVERITY_ORDER}
    for row in cosmos_results:
        if not isinstance(row, dict):
            continue
        severity = str(row.get("severity", "NONE")).upper()
        if severity not in counts:
            continue
        counts[severity] += 1
    return counts


def severity_ratios(cosmos_results: list[dict[str, Any]]) -> dict[str, float]:
    counts = severity_counts(cosmos_results)
    total = max(sum(counts.values()), 1)
    return {severity: counts[severity] / total for severity in SEVERITY_ORDER}


def confidence_values(cosmos_results: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in cosmos_results:
        if not isinstance(row, dict):
            continue
        values.append(_as_float(row.get("confidence", 0.0)))
    return values


def mean_signal_scores(candidates: list[dict[str, Any]]) -> dict[str, float]:
    means: dict[str, float] = {}
    for signal in SIGNAL_NAMES:
        signal_values: list[float] = []
        for row in candidates:
            if not isinstance(row, dict):
                continue
            scores = row.get("signal_scores", {})
            if not isinstance(scores, dict):
                continue
            signal_values.append(_as_float(scores.get(signal, 0.0)))
        means[signal] = sum(signal_values) / len(signal_values) if signal_values else 0.0
    return means


def positive_min_ttc_values(ttc_results: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in ttc_results:
        if not isinstance(row, dict):
            continue
        min_ttc = _as_float(row.get("min_ttc", -1.0), default=-1.0)
        if min_ttc > 0:
            values.append(min_ttc)
    return values
