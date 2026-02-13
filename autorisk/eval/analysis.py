"""Deep analysis module: signal contribution, error analysis, per-class metrics.

Provides systematic technical analysis beyond simple accuracy/F1 numbers,
demonstrating engineering rigor in understanding model behavior.
"""

from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import classification_report, precision_recall_fscore_support

from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)

SEVERITY_ORDER = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
LABELS = ["NONE", "LOW", "MEDIUM", "HIGH"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SignalAnalysisResult:
    """Per-signal contribution metrics."""
    signal_name: str
    spearman_rho: float = 0.0
    spearman_p: float = 1.0
    mean_score_by_severity: dict[str, float] = field(default_factory=dict)
    threshold_accuracy: float = 0.0
    threshold_f1: float = 0.0


@dataclass
class ErrorDetail:
    """Single misclassification detail."""
    clip_name: str
    gt_severity: str
    pred_severity: str
    error_type: str  # over_estimation, under_estimation
    severity_gap: int  # absolute difference in ordinal levels
    fused_score: float = 0.0
    reasoning_excerpt: str = ""


@dataclass
class PerClassMetrics:
    """Precision/recall/F1 per severity class."""
    label: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0


@dataclass
class AnalysisReport:
    """Complete analysis report aggregating all sub-analyses."""
    signal_analysis: list[SignalAnalysisResult] = field(default_factory=list)
    per_class_metrics: list[PerClassMetrics] = field(default_factory=list)
    error_details: list[ErrorDetail] = field(default_factory=list)
    error_summary: dict = field(default_factory=dict)
    # Signal-severity heatmap: signal_name -> severity -> mean_score
    signal_heatmap: dict[str, dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Analysis engine
# ---------------------------------------------------------------------------

class AnalysisEngine:
    """Run deep analysis on existing pipeline results."""

    def __init__(self) -> None:
        pass

    def run(
        self,
        cosmos_results_path: Path,
        candidates_csv_path: Path,
        gt_labels_path: Path,
    ) -> AnalysisReport:
        """Run full analysis suite.

        Args:
            cosmos_results_path: Path to cosmos_results.json.
            candidates_csv_path: Path to candidates.csv with per-signal scores.
            gt_labels_path: Path to gt_labels.csv.

        Returns:
            AnalysisReport with all sub-analyses.
        """
        # Load data
        cosmos_results = self._load_cosmos_results(cosmos_results_path)
        candidates = self._load_candidates(candidates_csv_path)
        gt_labels = self._load_gt_labels(gt_labels_path)

        report = AnalysisReport()

        # 1. Signal contribution analysis
        report.signal_analysis = self._analyze_signals(candidates, gt_labels)
        report.signal_heatmap = self._build_signal_heatmap(candidates, gt_labels)

        # 2. Per-class precision/recall/F1
        report.per_class_metrics = self._compute_per_class(cosmos_results, gt_labels)

        # 3. Error analysis
        report.error_details = self._analyze_errors(cosmos_results, gt_labels)
        report.error_summary = self._summarize_errors(report.error_details)

        log.info(
            "Analysis complete: %d signal results, %d per-class, %d errors",
            len(report.signal_analysis),
            len(report.per_class_metrics),
            len(report.error_details),
        )
        return report

    # -------------------------------------------------------------------
    # Signal contribution
    # -------------------------------------------------------------------

    def _analyze_signals(
        self,
        candidates: list[dict],
        gt_labels: dict[str, str],
    ) -> list[SignalAnalysisResult]:
        """Analyze per-signal contribution to severity detection."""
        signal_names = ["audio", "motion", "proximity", "fused"]
        results = []

        for signal in signal_names:
            scores = []
            gt_ordinals = []

            for cand in candidates:
                clip_name = Path(cand["clip_path"]).name
                gt_sev = self._match_gt(clip_name, gt_labels)
                if gt_sev is None:
                    continue

                if signal == "fused":
                    score = cand["fused_score"]
                else:
                    score = cand.get("signal_scores", {}).get(signal, 0.0)

                scores.append(score)
                gt_ordinals.append(SEVERITY_ORDER[gt_sev])

            if len(scores) < 3:
                results.append(SignalAnalysisResult(signal_name=signal))
                continue

            scores_arr = np.array(scores)
            gt_arr = np.array(gt_ordinals)

            # Spearman rank correlation
            rho, p_val = stats.spearmanr(scores_arr, gt_arr)

            # Mean score by severity
            mean_by_sev = {}
            for sev_name, sev_ord in SEVERITY_ORDER.items():
                mask = gt_arr == sev_ord
                if mask.sum() > 0:
                    mean_by_sev[sev_name] = float(scores_arr[mask].mean())

            # Threshold-based classification accuracy
            th_acc, th_f1 = self._threshold_classify(
                scores_arr, gt_arr, n_classes=4,
            )

            results.append(SignalAnalysisResult(
                signal_name=signal,
                spearman_rho=float(rho) if not np.isnan(rho) else 0.0,
                spearman_p=float(p_val) if not np.isnan(p_val) else 1.0,
                mean_score_by_severity=mean_by_sev,
                threshold_accuracy=th_acc,
                threshold_f1=th_f1,
            ))

            log.info(
                "  Signal %s: rho=%.3f (p=%.4f), thresh_acc=%.3f, thresh_f1=%.3f",
                signal, rho, p_val, th_acc, th_f1,
            )

        return results

    def _build_signal_heatmap(
        self,
        candidates: list[dict],
        gt_labels: dict[str, str],
    ) -> dict[str, dict[str, float]]:
        """Build signal x severity heatmap of mean scores."""
        signal_names = ["audio", "motion", "proximity"]
        heatmap: dict[str, dict[str, float]] = {}

        for signal in signal_names:
            sev_scores: dict[str, list[float]] = {s: [] for s in LABELS}

            for cand in candidates:
                clip_name = Path(cand["clip_path"]).name
                gt_sev = self._match_gt(clip_name, gt_labels)
                if gt_sev is None:
                    continue

                score = cand.get("signal_scores", {}).get(signal, 0.0)
                sev_scores[gt_sev].append(score)

            heatmap[signal] = {
                sev: float(np.mean(vals)) if vals else 0.0
                for sev, vals in sev_scores.items()
            }

        return heatmap

    @staticmethod
    def _threshold_classify(
        scores: np.ndarray,
        gt_ordinals: np.ndarray,
        n_classes: int = 4,
    ) -> tuple[float, float]:
        """Optimal threshold classification using quantile boundaries."""
        # Use GT distribution percentiles as thresholds
        sorted_scores = np.sort(scores)
        n = len(sorted_scores)

        # Count GT samples per class
        gt_counts = np.bincount(gt_ordinals, minlength=n_classes)
        cumulative = np.cumsum(gt_counts)
        thresholds = []
        for i in range(n_classes - 1):
            idx = min(cumulative[i], n - 1)
            thresholds.append(sorted_scores[idx])

        # Classify
        preds = np.zeros_like(gt_ordinals)
        for i, score in enumerate(scores):
            for j, th in enumerate(thresholds):
                if score <= th:
                    preds[i] = j
                    break
            else:
                preds[i] = n_classes - 1

        acc = float(np.mean(preds == gt_ordinals))
        f1 = float(
            precision_recall_fscore_support(
                gt_ordinals, preds, average="macro", zero_division=0,
            )[2]
        )
        return acc, f1

    # -------------------------------------------------------------------
    # Per-class metrics
    # -------------------------------------------------------------------

    def _compute_per_class(
        self,
        cosmos_results: list[dict],
        gt_labels: dict[str, str],
    ) -> list[PerClassMetrics]:
        """Compute precision, recall, F1 per severity class."""
        y_true, y_pred = [], []

        for entry in cosmos_results:
            clip_name = Path(entry.get("clip_path", "")).name
            gt_sev = self._match_gt(clip_name, gt_labels)
            if gt_sev is None:
                continue
            y_true.append(gt_sev)
            y_pred.append(entry.get("severity", "NONE"))

        if not y_true:
            return []

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, average=None, zero_division=0,
        )

        results = []
        for i, label in enumerate(LABELS):
            results.append(PerClassMetrics(
                label=label,
                precision=float(precision[i]),
                recall=float(recall[i]),
                f1=float(f1[i]),
                support=int(support[i]),
            ))
        return results

    # -------------------------------------------------------------------
    # Error analysis
    # -------------------------------------------------------------------

    def _analyze_errors(
        self,
        cosmos_results: list[dict],
        gt_labels: dict[str, str],
    ) -> list[ErrorDetail]:
        """Categorize every misclassification."""
        errors = []

        for entry in cosmos_results:
            clip_path = entry.get("clip_path", "")
            clip_name = Path(clip_path).name
            gt_sev = self._match_gt(clip_name, gt_labels)
            if gt_sev is None:
                continue

            pred_sev = entry.get("severity", "NONE")
            if gt_sev == pred_sev:
                continue

            gt_ord = SEVERITY_ORDER[gt_sev]
            pred_ord = SEVERITY_ORDER[pred_sev]
            gap = pred_ord - gt_ord

            errors.append(ErrorDetail(
                clip_name=clip_name,
                gt_severity=gt_sev,
                pred_severity=pred_sev,
                error_type="over_estimation" if gap > 0 else "under_estimation",
                severity_gap=abs(gap),
                fused_score=entry.get("fused_score", 0.0),
                reasoning_excerpt=entry.get("causal_reasoning", "")[:200],
            ))

        return errors

    @staticmethod
    def _summarize_errors(errors: list[ErrorDetail]) -> dict:
        """Produce aggregate error statistics."""
        if not errors:
            return {}

        n_total = len(errors)
        n_over = sum(1 for e in errors if e.error_type == "over_estimation")
        n_under = sum(1 for e in errors if e.error_type == "under_estimation")
        n_adjacent = sum(1 for e in errors if e.severity_gap == 1)
        n_major = sum(1 for e in errors if e.severity_gap >= 2)

        # Mining score correlation with error direction
        over_scores = [e.fused_score for e in errors if e.error_type == "over_estimation"]
        under_scores = [e.fused_score for e in errors if e.error_type == "under_estimation"]

        return {
            "total_errors": n_total,
            "over_estimation": n_over,
            "under_estimation": n_under,
            "adjacent_miss": n_adjacent,
            "major_miss": n_major,
            "over_estimation_pct": n_over / n_total * 100 if n_total else 0,
            "adjacent_miss_pct": n_adjacent / n_total * 100 if n_total else 0,
            "mean_mining_score_over": float(np.mean(over_scores)) if over_scores else 0.0,
            "mean_mining_score_under": float(np.mean(under_scores)) if under_scores else 0.0,
        }

    # -------------------------------------------------------------------
    # I/O helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _load_cosmos_results(path: Path) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_candidates(path: Path) -> list[dict]:
        candidates = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = {
                    "clip_path": row.get("clip_path", ""),
                    "fused_score": float(row.get("fused_score", 0.0)),
                    "peak_time_sec": float(row.get("peak_time_sec", 0.0)),
                }
                # Parse signal_scores dict from string
                ss_raw = row.get("signal_scores", "{}")
                try:
                    entry["signal_scores"] = ast.literal_eval(ss_raw)
                except (ValueError, SyntaxError):
                    entry["signal_scores"] = {}
                candidates.append(entry)
        return candidates

    @staticmethod
    def _load_gt_labels(path: Path) -> dict[str, str]:
        gt = {}
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clip = row.get("clip_path", "").strip()
                sev = row.get("severity", "NONE").strip().upper()
                if clip:
                    gt[Path(clip).name] = sev
        return gt

    @staticmethod
    def _match_gt(clip_name: str, gt_labels: dict[str, str]) -> str | None:
        """Match clip by filename."""
        return gt_labels.get(clip_name)

    # -------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------

    def save(self, report: AnalysisReport, output_dir: Path) -> Path:
        """Save analysis report to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "analysis_report.json"

        data = {
            "signal_analysis": [
                {
                    "signal_name": s.signal_name,
                    "spearman_rho": s.spearman_rho,
                    "spearman_p": s.spearman_p,
                    "mean_score_by_severity": s.mean_score_by_severity,
                    "threshold_accuracy": s.threshold_accuracy,
                    "threshold_f1": s.threshold_f1,
                }
                for s in report.signal_analysis
            ],
            "signal_heatmap": report.signal_heatmap,
            "per_class_metrics": [
                {
                    "label": m.label,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "support": m.support,
                }
                for m in report.per_class_metrics
            ],
            "error_summary": report.error_summary,
            "error_details": [
                {
                    "clip_name": e.clip_name,
                    "gt_severity": e.gt_severity,
                    "pred_severity": e.pred_severity,
                    "error_type": e.error_type,
                    "severity_gap": e.severity_gap,
                    "fused_score": e.fused_score,
                    "reasoning_excerpt": e.reasoning_excerpt,
                }
                for e in report.error_details
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        log.info("Saved analysis report to %s", path)
        return path
