"""Confidence calibration: Temperature Scaling, Reliability Diagram, ECE.

Evaluates how well the model's confidence scores correlate with actual
correctness, and applies post-hoc calibration via temperature scaling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)

SEVERITY_ORDER = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}


@dataclass
class CalibrationBin:
    """Single bin for reliability diagram."""
    bin_lower: float
    bin_upper: float
    avg_confidence: float
    avg_accuracy: float
    count: int
    gap: float  # |accuracy - confidence|


@dataclass
class CalibrationReport:
    """Full calibration analysis report."""
    # Expected Calibration Error
    ece: float = 0.0
    # Maximum Calibration Error
    mce: float = 0.0
    # Optimal temperature
    optimal_temperature: float = 1.0
    # ECE after temperature scaling
    ece_after: float = 0.0
    # Reliability diagram bins
    bins: list[CalibrationBin] = field(default_factory=list)
    # Bins after calibration
    bins_after: list[CalibrationBin] = field(default_factory=list)
    # Brier score (mean squared error of confidence vs correctness)
    brier_score: float = 0.0
    brier_score_after: float = 0.0
    # Per-severity confidence stats
    confidence_by_severity: dict[str, dict] = field(default_factory=dict)
    n_samples: int = 0


def _compute_bins(
    confidences: np.ndarray,
    corrects: np.ndarray,
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Compute reliability diagram bins."""
    bins = []
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (confidences >= lo) & (confidences <= hi)

        count = mask.sum()
        if count == 0:
            bins.append(CalibrationBin(
                bin_lower=lo, bin_upper=hi,
                avg_confidence=0.0, avg_accuracy=0.0,
                count=0, gap=0.0,
            ))
            continue

        avg_conf = float(confidences[mask].mean())
        avg_acc = float(corrects[mask].mean())

        bins.append(CalibrationBin(
            bin_lower=lo, bin_upper=hi,
            avg_confidence=avg_conf, avg_accuracy=avg_acc,
            count=int(count), gap=abs(avg_acc - avg_conf),
        ))

    return bins


def _compute_ece(bins: list[CalibrationBin], n_total: int) -> float:
    """Compute Expected Calibration Error from bins."""
    if n_total == 0:
        return 0.0
    ece = sum(b.count * b.gap for b in bins) / n_total
    return float(ece)


def _compute_mce(bins: list[CalibrationBin]) -> float:
    """Compute Maximum Calibration Error."""
    non_empty = [b for b in bins if b.count > 0]
    if not non_empty:
        return 0.0
    return float(max(b.gap for b in non_empty))


def _temperature_scale(confidences: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to confidence scores.

    For a single-value confidence (not logits), we use the formula:
    scaled = conf^(1/T) / (conf^(1/T) + (1-conf)^(1/T))
    """
    eps = 1e-8
    c = np.clip(confidences, eps, 1 - eps)

    if abs(temperature - 1.0) < eps:
        return c

    inv_t = 1.0 / temperature
    c_scaled = c ** inv_t
    nc_scaled = (1 - c) ** inv_t
    return c_scaled / (c_scaled + nc_scaled)


def _find_optimal_temperature(
    confidences: np.ndarray,
    corrects: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Find temperature that minimizes ECE."""
    def objective(t):
        scaled = _temperature_scale(confidences, t)
        bins = _compute_bins(scaled, corrects, n_bins)
        return _compute_ece(bins, len(confidences))

    result = minimize_scalar(objective, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


class CalibrationAnalyzer:
    """Analyze and calibrate model confidence scores."""

    def run(
        self,
        cosmos_results_path: Path,
        gt_labels_path: Path,
        n_bins: int = 10,
    ) -> CalibrationReport:
        """Run full calibration analysis.

        Args:
            cosmos_results_path: Path to cosmos_results.json.
            gt_labels_path: Path to gt_labels.csv.
            n_bins: Number of bins for reliability diagram.

        Returns:
            CalibrationReport with ECE, MCE, temperature, bins.
        """
        import csv

        # Load data
        with open(cosmos_results_path, encoding="utf-8") as f:
            results = json.load(f)

        gt_labels = {}
        with open(gt_labels_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clip = row.get("clip_path", "").strip()
                sev = row.get("severity", "NONE").strip().upper()
                if clip:
                    gt_labels[Path(clip).name] = sev

        # Build arrays
        confidences = []
        corrects = []
        pred_sevs = []
        gt_sevs = []

        for entry in results:
            clip_name = Path(entry.get("clip_path", "")).name
            gt_sev = gt_labels.get(clip_name)
            if gt_sev is None:
                continue
            if not entry.get("parse_success", True):
                continue

            pred_sev = entry.get("severity", "NONE")
            conf = entry.get("confidence", 0.5)

            confidences.append(conf)
            corrects.append(1.0 if pred_sev == gt_sev else 0.0)
            pred_sevs.append(pred_sev)
            gt_sevs.append(gt_sev)

        if len(confidences) < 3:
            log.warning("Calibration: too few samples (%d)", len(confidences))
            return CalibrationReport(n_samples=len(confidences))

        conf_arr = np.array(confidences)
        correct_arr = np.array(corrects)

        # Pre-calibration
        bins_before = _compute_bins(conf_arr, correct_arr, n_bins)
        ece_before = _compute_ece(bins_before, len(conf_arr))
        mce_before = _compute_mce(bins_before)
        brier_before = float(np.mean((conf_arr - correct_arr) ** 2))

        # Find optimal temperature
        opt_temp = _find_optimal_temperature(conf_arr, correct_arr, n_bins)

        # Post-calibration
        scaled_conf = _temperature_scale(conf_arr, opt_temp)
        bins_after = _compute_bins(scaled_conf, correct_arr, n_bins)
        ece_after = _compute_ece(bins_after, len(scaled_conf))
        brier_after = float(np.mean((scaled_conf - correct_arr) ** 2))

        # Per-severity confidence stats
        conf_by_sev: dict[str, dict] = {}
        for sev in ["NONE", "LOW", "MEDIUM", "HIGH"]:
            mask = [i for i, s in enumerate(gt_sevs) if s == sev]
            if not mask:
                continue
            sev_confs = conf_arr[mask]
            sev_correct = correct_arr[mask]
            conf_by_sev[sev] = {
                "mean_confidence": float(sev_confs.mean()),
                "std_confidence": float(sev_confs.std()),
                "accuracy": float(sev_correct.mean()),
                "n_samples": len(mask),
                "overconfident": float(sev_confs.mean()) > float(sev_correct.mean()),
            }

        report = CalibrationReport(
            ece=ece_before,
            mce=mce_before,
            optimal_temperature=opt_temp,
            ece_after=ece_after,
            bins=bins_before,
            bins_after=bins_after,
            brier_score=brier_before,
            brier_score_after=brier_after,
            confidence_by_severity=conf_by_sev,
            n_samples=len(confidences),
        )

        log.info(
            "Calibration: ECE=%.4f→%.4f (T=%.2f), Brier=%.4f→%.4f",
            ece_before, ece_after, opt_temp, brier_before, brier_after,
        )

        return report

    @staticmethod
    def save(report: CalibrationReport, output_dir: Path) -> Path:
        """Save calibration report to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "calibration_report.json"

        data = {
            "ece": report.ece,
            "mce": report.mce,
            "optimal_temperature": report.optimal_temperature,
            "ece_after": report.ece_after,
            "brier_score": report.brier_score,
            "brier_score_after": report.brier_score_after,
            "n_samples": report.n_samples,
            "confidence_by_severity": report.confidence_by_severity,
            "bins": [
                {
                    "bin_lower": b.bin_lower,
                    "bin_upper": b.bin_upper,
                    "avg_confidence": b.avg_confidence,
                    "avg_accuracy": b.avg_accuracy,
                    "count": b.count,
                    "gap": b.gap,
                }
                for b in report.bins
            ],
            "bins_after": [
                {
                    "bin_lower": b.bin_lower,
                    "bin_upper": b.bin_upper,
                    "avg_confidence": b.avg_confidence,
                    "avg_accuracy": b.avg_accuracy,
                    "count": b.count,
                    "gap": b.gap,
                }
                for b in report.bins_after
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        log.info("Calibration: Saved report to %s", path)
        return path

    @staticmethod
    def load(path: Path) -> CalibrationReport:
        """Load calibration report from JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return CalibrationReport(
            ece=data["ece"],
            mce=data["mce"],
            optimal_temperature=data["optimal_temperature"],
            ece_after=data["ece_after"],
            brier_score=data["brier_score"],
            brier_score_after=data["brier_score_after"],
            n_samples=data["n_samples"],
            confidence_by_severity=data.get("confidence_by_severity", {}),
            bins=[
                CalibrationBin(**b) for b in data.get("bins", [])
            ],
            bins_after=[
                CalibrationBin(**b) for b in data.get("bins_after", [])
            ],
        )
