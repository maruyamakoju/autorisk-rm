"""Main evaluator: loads GT, computes metrics, produces failure analysis."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from autorisk.cosmos.schema import CosmosResponse
from autorisk.eval.checklist import ExplanationChecklist
from autorisk.eval.metrics import compute_accuracy, compute_macro_f1
from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)


@dataclass
class EvalReport:
    """Aggregated evaluation report."""
    n_samples: int = 0
    accuracy: float = 0.0
    macro_f1: float = 0.0
    checklist_means: dict[str, float] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)


def _build_gt_indices(
    gt_labels: dict[str, str],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Build multiple lookup indices for flexible GT matching.

    Returns:
        Tuple of (gt_by_path, gt_by_name, gt_by_time).
    """
    gt_by_name: dict[str, str] = {}
    gt_by_time: dict[str, str] = {}
    for k, v in gt_labels.items():
        gt_by_name[Path(k).name] = v
        t_match = re.search(r"_t([\d.]+)s", Path(k).name)
        if t_match:
            gt_by_time[t_match.group(1)] = v
    return gt_labels, gt_by_name, gt_by_time


def _match_gt_label(
    clip_path: str,
    gt_by_path: dict[str, str],
    gt_by_name: dict[str, str],
    gt_by_time: dict[str, str],
) -> str | None:
    """Match a clip path to its GT label using multiple strategies.

    Tries: exact path → filename → timestamp extraction.
    """
    gt_sev = gt_by_path.get(clip_path)
    if gt_sev is not None:
        return gt_sev
    gt_sev = gt_by_name.get(Path(clip_path).name)
    if gt_sev is not None:
        return gt_sev
    t_match = re.search(r"_t([\d.]+)s", Path(clip_path).name)
    if t_match:
        return gt_by_time.get(t_match.group(1))
    return None


class Evaluator:
    """End-to-end evaluator: GT loading, classification metrics, checklist."""

    def __init__(self, severity_labels: list[str] | None = None) -> None:
        self.labels = severity_labels or ["NONE", "LOW", "MEDIUM", "HIGH"]
        self.checklist = ExplanationChecklist()

    def load_gt_labels(self, gt_path: str | Path) -> dict[str, str]:
        """Load ground-truth severity labels from CSV.

        Expected CSV format: clip_path,severity

        Returns:
            Dict mapping clip_path -> severity string.
        """
        gt: dict[str, str] = {}
        gt_path = Path(gt_path)
        if not gt_path.exists():
            log.warning("GT file not found: %s", gt_path)
            return gt

        with open(gt_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clip = row.get("clip_path", "").strip()
                sev = row.get("severity", "NONE").strip().upper()
                if clip:
                    gt[clip] = sev

        log.info("Loaded %d GT labels from %s", len(gt), gt_path)
        return gt

    def load_checklist_gt(
        self, checklist_path: str | Path,
    ) -> dict[str, dict[str, int]]:
        """Load checklist GT from CSV.

        Expected format: clip_path,actors_accurate,causal_clear,...

        Returns:
            Dict mapping clip_path -> {item: score}.
        """
        gt_map: dict[str, dict[str, int]] = {}
        path = Path(checklist_path)
        if not path.exists():
            log.warning("Checklist GT not found: %s", path)
            return gt_map

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clip = row.get("clip_path", "").strip()
                if not clip:
                    continue
                scores = {}
                for key in [
                    "actors_accurate", "causal_clear", "spatial_specific",
                    "prediction_plausible", "action_reasonable",
                ]:
                    scores[key] = int(row.get(key, 0))
                gt_map[clip] = scores

        log.info("Loaded checklist GT for %d clips", len(gt_map))
        return gt_map

    def evaluate(
        self,
        responses: list[CosmosResponse],
        gt_labels: dict[str, str] | None = None,
        checklist_gt: dict[str, dict[str, int]] | None = None,
    ) -> EvalReport:
        """Run full evaluation pipeline.

        Args:
            responses: Cosmos inference responses.
            gt_labels: Optional GT severity labels.
            checklist_gt: Optional GT checklist scores.

        Returns:
            EvalReport with all metrics.
        """
        report = EvalReport(n_samples=len(responses))

        # Classification metrics (only if GT available)
        if gt_labels:
            gt_by_path, gt_by_name, gt_by_time = _build_gt_indices(gt_labels)

            y_true, y_pred = [], []
            for r in responses:
                gt_sev = _match_gt_label(
                    r.request.clip_path, gt_by_path, gt_by_name, gt_by_time,
                )
                if gt_sev is not None:
                    y_true.append(gt_sev)
                    y_pred.append(r.assessment.severity)

            if y_true:
                report.accuracy = compute_accuracy(y_true, y_pred)
                report.macro_f1 = compute_macro_f1(y_true, y_pred, labels=self.labels)
                report.confusion = self._confusion_matrix(y_true, y_pred)
                report.failures = self._failure_analysis(responses, gt_labels)

                log.info(
                    "Classification: accuracy=%.3f, macro_f1=%.3f (%d samples)",
                    report.accuracy, report.macro_f1, len(y_true),
                )

        # Checklist evaluation (MEDIUM/HIGH only for meaningful scores)
        checklist_results = self.checklist.evaluate_batch(
            responses, checklist_gt, severity_filter={"MEDIUM", "HIGH"},
        )
        if checklist_results:
            report.checklist_means = ExplanationChecklist.aggregate(checklist_results)
            log.info(
                "Checklist mean total: %.2f/5 (%d MEDIUM/HIGH clips)",
                report.checklist_means.get("mean_total", 0),
                len(checklist_results),
            )
        else:
            # Fallback: evaluate all clips if no MEDIUM/HIGH exist
            checklist_results = self.checklist.evaluate_batch(
                responses, checklist_gt, severity_filter=set(),
            )
            report.checklist_means = ExplanationChecklist.aggregate(checklist_results)
            log.info(
                "Checklist mean total: %.2f/5 (all %d clips, no MEDIUM/HIGH found)",
                report.checklist_means.get("mean_total", 0),
                len(checklist_results),
            )

        return report

    def _confusion_matrix(
        self, y_true: list[str], y_pred: list[str],
    ) -> dict[str, dict[str, int]]:
        """Build a confusion matrix as nested dicts."""
        matrix: dict[str, dict[str, int]] = {
            label: {l: 0 for l in self.labels} for label in self.labels
        }
        for t, p in zip(y_true, y_pred):
            if t in matrix and p in matrix[t]:
                matrix[t][p] += 1
        return matrix

    def _failure_analysis(
        self,
        responses: list[CosmosResponse],
        gt_labels: dict[str, str],
    ) -> list[dict]:
        """Identify misclassified examples."""
        gt_by_path, gt_by_name, gt_by_time = _build_gt_indices(gt_labels)

        failures = []
        for r in responses:
            gt_sev = _match_gt_label(
                r.request.clip_path, gt_by_path, gt_by_name, gt_by_time,
            )
            if gt_sev is None:
                continue
            pred_sev = r.assessment.severity
            if gt_sev != pred_sev:
                failures.append({
                    "clip_path": r.request.clip_path,
                    "gt_severity": gt_sev,
                    "pred_severity": pred_sev,
                    "causal_reasoning": r.assessment.causal_reasoning[:200],
                    "confidence": r.assessment.confidence,
                })
        return failures

    def save_report(self, report: EvalReport, output_dir: Path) -> Path:
        """Save evaluation report as JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "eval_report.json"

        data = {
            "n_samples": report.n_samples,
            "accuracy": report.accuracy,
            "macro_f1": report.macro_f1,
            "checklist_means": report.checklist_means,
            "confusion_matrix": report.confusion,
            "n_failures": len(report.failures),
            "failures": report.failures,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        log.info("Saved eval report to %s", path)
        return path
