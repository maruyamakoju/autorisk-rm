"""Data loading utilities for the Streamlit dashboard."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

try:
    import streamlit as st
except Exception:  # pragma: no cover - optional dependency fallback
    class _StreamlitFallback:
        @staticmethod
        def cache_data(func=None, **_kwargs):
            if func is None:
                def _decorator(inner):
                    return inner
                return _decorator
            return func

    st = _StreamlitFallback()  # type: ignore[assignment]

# Import centralized constants from comparison_utils
# Note: This must come after streamlit import attempt
try:
    from autorisk.dashboard.comparison_utils import SEVERITY_COLORS, SEVERITY_ORDER
except ImportError:
    # Fallback if comparison_utils not available
    SEVERITY_ORDER = ["NONE", "LOW", "MEDIUM", "HIGH"]
    SEVERITY_COLORS = {
        "NONE": "#6B7280",
        "LOW": "#10B981",
        "MEDIUM": "#F59E0B",
        "HIGH": "#EF4444",
    }

SEVERITY_EMOJI = {
    "NONE": "",
    "LOW": "",
    "MEDIUM": "",
    "HIGH": "",
}


def _safe_load_json(path: Path) -> dict | list | None:
    if not path.exists() or not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _warn(warnings: list[str], message: str) -> None:
    warnings.append(message[:300])


def _parse_int(raw: str | None, *, field: str, row_no: int, warnings: list[str]) -> int | None:
    try:
        return int(str(raw))
    except Exception:
        _warn(warnings, f"candidates.csv row {row_no}: invalid int field '{field}' = {raw!r}")
        return None


def _parse_float(raw: str | None, *, field: str, row_no: int, warnings: list[str]) -> float | None:
    try:
        return float(str(raw))
    except Exception:
        _warn(warnings, f"candidates.csv row {row_no}: invalid float field '{field}' = {raw!r}")
        return None


def _parse_signal_scores(raw: str | None, *, row_no: int, warnings: list[str]) -> dict:
    text = str(raw or "").strip()
    if text == "":
        return {}
    loaded = None
    try:
        loaded = ast.literal_eval(text)
    except Exception:
        try:
            loaded = json.loads(text)
        except Exception:
            _warn(warnings, f"candidates.csv row {row_no}: invalid signal_scores; using empty dict")
            return {}
    if not isinstance(loaded, dict):
        _warn(warnings, f"candidates.csv row {row_no}: signal_scores is not a dict; using empty dict")
        return {}
    normalized: dict[str, float] = {}
    for k, v in loaded.items():
        try:
            normalized[str(k)] = float(v)
        except Exception:
            _warn(warnings, f"candidates.csv row {row_no}: signal_scores[{k!r}] is non-numeric; dropped")
    return normalized


@st.cache_data
def load_data(run_dir: str) -> dict:
    """Load all pipeline result data from a run directory."""
    run = Path(run_dir)
    data: dict = {}
    warnings: list[str] = []

    # Core results
    def _load_json_with_warning(name: str, fallback):
        path = run / name
        try:
            loaded = _safe_load_json(path)
        except Exception as exc:
            _warn(warnings, f"{name}: failed to parse JSON ({str(exc)[:120]})")
            return fallback
        return loaded if loaded is not None else fallback

    data["cosmos_results"] = _load_json_with_warning("cosmos_results.json", [])
    data["eval_report"] = _load_json_with_warning("eval_report.json", {})
    data["analysis_report"] = _load_json_with_warning("analysis_report.json", {})
    data["ablation_results"] = _load_json_with_warning("ablation_results.json", [])
    data["ttc_results"] = _load_json_with_warning("ttc_results.json", [])
    data["grounding_report"] = _load_json_with_warning("grounding_report.json", {})
    data["calibration_report"] = _load_json_with_warning("calibration_report.json", {})
    data["saliency_results"] = _load_json_with_warning("saliency_results.json", [])
    data["saliency_images"] = _load_json_with_warning("saliency_images.json", {})

    # Enhanced correction results
    correction_dir = run.parent / "enhanced_correction"
    data["correction_report"] = _safe_load_json(correction_dir / "eval_report.json") if correction_dir.exists() else None
    data["correction_results"] = _safe_load_json(correction_dir / "corrected_results.json") if correction_dir.exists() else None

    # Predict 2 results
    predict_dir = run / "predictions"
    data["predict_results"] = _safe_load_json(predict_dir / "predict_results.json") if predict_dir.exists() else None

    # Candidates CSV
    candidates = []
    csv_path = run / "candidates.csv"
    if csv_path.exists():
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row_no, row in enumerate(csv.DictReader(f), start=2):
                    rank = _parse_int(row.get("rank"), field="rank", row_no=row_no, warnings=warnings)
                    peak_time_sec = _parse_float(
                        row.get("peak_time_sec"),
                        field="peak_time_sec",
                        row_no=row_no,
                        warnings=warnings,
                    )
                    start_sec = _parse_float(row.get("start_sec"), field="start_sec", row_no=row_no, warnings=warnings)
                    end_sec = _parse_float(row.get("end_sec"), field="end_sec", row_no=row_no, warnings=warnings)
                    fused_score = _parse_float(
                        row.get("fused_score"),
                        field="fused_score",
                        row_no=row_no,
                        warnings=warnings,
                    )
                    if None in {rank, peak_time_sec, start_sec, end_sec, fused_score}:
                        continue
                    parsed = dict(row)
                    parsed["rank"] = rank
                    parsed["peak_time_sec"] = peak_time_sec
                    parsed["start_sec"] = start_sec
                    parsed["end_sec"] = end_sec
                    parsed["fused_score"] = fused_score
                    parsed["signal_scores"] = _parse_signal_scores(
                        row.get("signal_scores"),
                        row_no=row_no,
                        warnings=warnings,
                    )
                    candidates.append(parsed)
        except Exception as exc:
            _warn(warnings, f"candidates.csv: failed to parse ({str(exc)[:120]})")
    data["candidates"] = candidates

    # GT labels
    gt: dict[str, str] = {}
    gt_path = run.parent.parent / "data" / "annotations" / "gt_labels.csv"
    if not gt_path.exists():
        gt_path = run / "gt_labels.csv"
    if gt_path.exists():
        try:
            with open(gt_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    clip_name = Path(str(row.get("clip_path", ""))).name
                    if clip_name == "":
                        continue
                    gt[clip_name] = str(row.get("severity", "NONE"))
        except Exception as exc:
            _warn(warnings, f"{gt_path.name}: failed to parse ({str(exc)[:120]})")
    data["gt_labels"] = gt

    # Lookup helpers
    data["clips_dir"] = str(run / "clips")
    data["frames_dir"] = str(run / "frames")

    # Build per-clip lookup
    ttc_lookup: dict[str, dict] = {}
    for t in data["ttc_results"] if isinstance(data["ttc_results"], list) else []:
        if not isinstance(t, dict):
            continue
        name = Path(str(t.get("clip_path", ""))).name
        if name == "":
            continue
        ttc_lookup[name] = t
    data["ttc_lookup"] = ttc_lookup

    grounding_lookup: dict[str, dict] = {}
    details = data.get("grounding_report", {}).get("details", [])
    if isinstance(details, list):
        for g in details:
            if not isinstance(g, dict):
                continue
            clip_name = str(g.get("clip_name", ""))
            if clip_name == "":
                continue
            grounding_lookup[clip_name] = g
    data["grounding_lookup"] = grounding_lookup
    data["load_warnings"] = warnings
    data["load_warnings_count"] = len(warnings)

    return data
