"""Data loading utilities for the Streamlit dashboard."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import streamlit as st

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
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_data(run_dir: str) -> dict:
    """Load all pipeline result data from a run directory."""
    run = Path(run_dir)
    data: dict = {}

    # Core results
    data["cosmos_results"] = _safe_load_json(run / "cosmos_results.json") or []
    data["eval_report"] = _safe_load_json(run / "eval_report.json") or {}
    data["analysis_report"] = _safe_load_json(run / "analysis_report.json") or {}
    data["ablation_results"] = _safe_load_json(run / "ablation_results.json") or []
    data["ttc_results"] = _safe_load_json(run / "ttc_results.json") or []
    data["grounding_report"] = _safe_load_json(run / "grounding_report.json") or {}
    data["calibration_report"] = _safe_load_json(run / "calibration_report.json") or {}
    data["saliency_results"] = _safe_load_json(run / "saliency_results.json") or []
    data["saliency_images"] = _safe_load_json(run / "saliency_images.json") or {}

    # Candidates CSV
    candidates = []
    csv_path = run / "candidates.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["rank"] = int(row["rank"])
                row["peak_time_sec"] = float(row["peak_time_sec"])
                row["start_sec"] = float(row["start_sec"])
                row["end_sec"] = float(row["end_sec"])
                row["fused_score"] = float(row["fused_score"])
                row["signal_scores"] = ast.literal_eval(row["signal_scores"])
                candidates.append(row)
    data["candidates"] = candidates

    # GT labels
    gt: dict[str, str] = {}
    gt_path = run.parent.parent / "data" / "annotations" / "gt_labels.csv"
    if not gt_path.exists():
        gt_path = run / "gt_labels.csv"
    if gt_path.exists():
        with open(gt_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                clip_name = Path(row["clip_path"]).name
                gt[clip_name] = row["severity"]
    data["gt_labels"] = gt

    # Lookup helpers
    data["clips_dir"] = str(run / "clips")
    data["frames_dir"] = str(run / "frames")

    # Build per-clip lookup
    ttc_lookup: dict[str, dict] = {}
    for t in data["ttc_results"]:
        name = Path(t["clip_path"]).name
        ttc_lookup[name] = t
    data["ttc_lookup"] = ttc_lookup

    grounding_lookup: dict[str, dict] = {}
    for g in data.get("grounding_report", {}).get("details", []):
        grounding_lookup[g["clip_name"]] = g
    data["grounding_lookup"] = grounding_lookup

    return data
