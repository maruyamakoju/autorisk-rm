"""Enhanced signal-based severity correction.

Uses TTC, fused signal scores, and critical object counts to post-correct
Cosmos VLM severity predictions via a 6-rule system optimized with LOOCV.
"""

from __future__ import annotations

import copy
import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from autorisk.utils.logger import setup_logger

log = setup_logger(__name__)

SEVERITY_ORDER = ["NONE", "LOW", "MEDIUM", "HIGH"]
_SEV_IDX = {s: i for i, s in enumerate(SEVERITY_ORDER)}

DEFAULT_PARAMS: dict[str, float] = {
    "ttc_safe": 1.0,
    "fused_none": 0.55,
    "n_crit_none": 1,
    "ttc_high": 0.25,
    "n_crit_high": 4,
    "fused_med": 0.65,
    "ttc_med": 0.6,
    "ttc_high_down": 1.5,
    "n_crit_high_down": 2,
    "fused_med_down": 0.5,
    "ttc_med_down": 0.6,
    "ttc_none_up": 0.5,
    "conf_gate": 1.0,
}

# Search ranges for grid/random search
PARAM_RANGES: dict[str, tuple[float, float, float]] = {
    # (min, max, step)
    "ttc_safe": (0.6, 1.6, 0.1),
    "fused_none": (0.35, 0.65, 0.05),
    "n_crit_none": (0, 3, 1),
    "ttc_high": (0.15, 0.45, 0.05),
    "n_crit_high": (2, 6, 1),
    "fused_med": (0.55, 0.80, 0.05),
    "ttc_med": (0.3, 0.8, 0.1),
    "ttc_high_down": (1.0, 2.0, 0.1),
    "n_crit_high_down": (1, 4, 1),
    "fused_med_down": (0.35, 0.60, 0.05),
    "ttc_med_down": (0.4, 0.8, 0.1),
    "ttc_none_up": (0.3, 0.7, 0.1),
    "conf_gate": (0.5, 1.0, 0.1),
}


@dataclass
class CorrectionResult:
    clip: str
    gt: str
    pred: str
    original: str
    reason: str | None
    correct: bool


@dataclass
class CorrectionReport:
    accuracy: float
    macro_f1: float
    n_correct: int
    n_total: int
    confusion_matrix: list[list[int]]
    per_class: dict[str, dict]
    details: list[dict]


class EnhancedCorrector:
    """Apply 6-rule signal-based correction to VLM severity predictions."""

    def __init__(self, params: dict[str, float] | None = None) -> None:
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)

    def correct_single(
        self,
        severity: str,
        fused_score: float,
        min_ttc: float,
        n_critical: int,
        confidence: float = 0.0,
    ) -> tuple[str, str | None]:
        """Apply correction rules to a single clip.

        Returns:
            (corrected_severity, reason_string_or_None)
        """
        p = self.params

        # Confidence gate: skip correction if confidence exceeds gate
        if confidence > p["conf_gate"]:
            return severity, None

        # R1: NONE detection - safe TTC + low fused + few critical objects
        if (
            min_ttc >= p["ttc_safe"]
            and fused_score < p["fused_none"]
            and n_critical <= p["n_crit_none"]
        ):
            reason = f"R1:NONE (TTC={min_ttc:.1f},fused={fused_score:.2f},ncrit={n_critical})"
            return "NONE", reason

        # R2: HIGH detection - very low TTC + many critical objects
        if min_ttc <= p["ttc_high"] and n_critical >= p["n_crit_high"]:
            reason = f"R2:HIGH (TTC={min_ttc:.2f},ncrit={n_critical})"
            return "HIGH", reason

        # R3: MEDIUM promotion - high fused + low TTC
        if (
            severity == "LOW"
            and fused_score >= p["fused_med"]
            and min_ttc <= p["ttc_med"]
        ):
            reason = f"R3:MED (fused={fused_score:.2f},TTC={min_ttc:.1f})"
            return "MEDIUM", reason

        # R4: HIGH demotion - safe TTC + few critical
        if (
            severity == "HIGH"
            and min_ttc >= p["ttc_high_down"]
            and n_critical <= p["n_crit_high_down"]
        ):
            reason = f"R4:MED (HIGH downgrade: TTC={min_ttc:.1f},ncrit={n_critical})"
            return "MEDIUM", reason

        # R5: MEDIUM demotion - low fused + safe TTC
        if (
            severity == "MEDIUM"
            and fused_score < p["fused_med_down"]
            and min_ttc >= p["ttc_med_down"]
        ):
            reason = f"R5:LOW (weak for MED: fused={fused_score:.2f},TTC={min_ttc:.1f})"
            return "LOW", reason

        # R6: NONE upgrade - low TTC suggests some danger
        if severity == "NONE" and min_ttc <= p["ttc_none_up"]:
            reason = f"R6:LOW (NONE upgrade: TTC={min_ttc:.1f},ncrit={n_critical})"
            return "LOW", reason

        return severity, None

    def correct_batch(
        self,
        cosmos_results: list[dict],
        ttc_data: list[dict],
        candidates: list[dict] | None = None,
    ) -> list[dict]:
        """Apply correction to a batch of results.

        Args:
            cosmos_results: List of cosmos result dicts (with severity, clip_path, etc.)
            ttc_data: List of TTC result dicts (with clip_path, min_ttc, n_critical)
            candidates: Optional list of candidate dicts (with fused_score)

        Returns:
            Corrected cosmos results (deep copies with original_severity/correction_reason added)
        """
        # Build lookups by clip name
        ttc_lookup = {}
        for t in ttc_data:
            name = Path(str(t.get("clip_path", ""))).name
            if name:
                ttc_lookup[name] = t

        cand_lookup = {}
        if candidates:
            for c in candidates:
                cp = c.get("clip_path", "")
                name = Path(str(cp)).name if cp else ""
                if name:
                    cand_lookup[name] = c

        corrected = []
        for r in cosmos_results:
            out = copy.deepcopy(r)
            clip_name = Path(str(r.get("clip_path", ""))).name
            original_sev = r.get("severity", "NONE")

            # Get TTC data
            ttc = ttc_lookup.get(clip_name, {})
            min_ttc = ttc.get("min_ttc", float("inf"))
            if min_ttc < 0:
                min_ttc = float("inf")
            n_critical = ttc.get("n_critical", 0)

            # Get fused score from candidate or result
            fused = r.get("fused_score", 0.0)
            if clip_name in cand_lookup:
                fused = cand_lookup[clip_name].get("fused_score", fused)

            confidence = r.get("confidence", 0.0)

            new_sev, reason = self.correct_single(
                severity=original_sev,
                fused_score=fused,
                min_ttc=min_ttc,
                n_critical=n_critical,
                confidence=confidence,
            )

            out["severity"] = new_sev
            out["original_severity"] = original_sev
            out["correction_reason"] = reason

            corrected.append(out)

        return corrected


def evaluate_enhanced(
    corrected_results: list[dict],
    gt_labels: dict[str, str],
) -> CorrectionReport:
    """Evaluate corrected results against ground truth.

    Args:
        corrected_results: List of corrected result dicts
        gt_labels: Dict mapping clip_name -> severity string

    Returns:
        CorrectionReport with accuracy, F1, confusion matrix, details
    """
    n_classes = len(SEVERITY_ORDER)
    cm = [[0] * n_classes for _ in range(n_classes)]
    details: list[dict] = []
    n_correct = 0
    n_total = 0

    for r in corrected_results:
        clip_name = Path(str(r.get("clip_path", ""))).name
        if clip_name not in gt_labels:
            continue

        gt_sev = gt_labels[clip_name]
        pred_sev = r.get("severity", "NONE")
        original_sev = r.get("original_severity", pred_sev)
        reason = r.get("correction_reason")

        if gt_sev not in _SEV_IDX or pred_sev not in _SEV_IDX:
            continue

        gt_idx = _SEV_IDX[gt_sev]
        pred_idx = _SEV_IDX[pred_sev]
        cm[gt_idx][pred_idx] += 1

        correct = gt_sev == pred_sev
        if correct:
            n_correct += 1
        n_total += 1

        details.append({
            "clip": clip_name,
            "gt": gt_sev,
            "pred": pred_sev,
            "original": original_sev,
            "reason": reason,
            "correct": correct,
        })

    accuracy = n_correct / max(n_total, 1)

    # Per-class metrics
    per_class: dict[str, dict] = {}
    for i, sev in enumerate(SEVERITY_ORDER):
        tp = cm[i][i]
        per_class[sev] = {
            "count": sum(cm[i]),
            "correct": tp,
            "recall": tp / max(sum(cm[i]), 1),
        }

    # Macro-F1 via sklearn for consistency with saved outputs
    try:
        from sklearn.metrics import f1_score as _f1

        gt_list = [d["gt"] for d in details]
        pred_list = [d["pred"] for d in details]
        macro_f1 = float(_f1(gt_list, pred_list, average="macro", labels=SEVERITY_ORDER))
    except ImportError:
        # Fallback: manual macro-F1
        f1_scores = []
        for i in range(n_classes):
            tp = cm[i][i]
            fn = sum(cm[i]) - tp
            fp = sum(cm[j][i] for j in range(n_classes)) - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_scores.append(f1)
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return CorrectionReport(
        accuracy=accuracy,
        macro_f1=macro_f1,
        n_correct=n_correct,
        n_total=n_total,
        confusion_matrix=cm,
        per_class=per_class,
        details=details,
    )


def _random_params() -> dict[str, float]:
    """Generate random parameter set within defined ranges."""
    params: dict[str, float] = {}
    for key, (lo, hi, step) in PARAM_RANGES.items():
        n_steps = int((hi - lo) / step) + 1
        params[key] = lo + random.randint(0, n_steps - 1) * step
    return params


def grid_search_enhanced(
    cosmos_results: list[dict],
    ttc_data: list[dict],
    gt_labels: dict[str, str],
    candidates: list[dict] | None = None,
    n_random: int = 5000,
    seed: int = 42,
) -> tuple[dict[str, float], CorrectionReport]:
    """Random search for best correction parameters.

    Returns:
        (best_params, best_report)
    """
    random.seed(seed)
    best_acc = -1.0
    best_params = dict(DEFAULT_PARAMS)
    best_report: CorrectionReport | None = None

    for _ in range(n_random):
        params = _random_params()
        corrector = EnhancedCorrector(params)
        corrected = corrector.correct_batch(cosmos_results, ttc_data, candidates)
        report = evaluate_enhanced(corrected, gt_labels)

        if report.accuracy > best_acc:
            best_acc = report.accuracy
            best_params = params
            best_report = report

    log.info("Grid search best accuracy: %.3f", best_acc)
    return best_params, best_report  # type: ignore[return-value]


def loocv_grid_search(
    cosmos_results: list[dict],
    ttc_data: list[dict],
    gt_labels: dict[str, str],
    candidates: list[dict] | None = None,
    n_random: int = 5000,
    seed: int = 42,
) -> dict:
    """Leave-One-Out Cross-Validation grid search.

    For each fold, hold out one clip, optimize on the rest, predict held-out clip.

    Returns:
        LOOCV report dict with accuracy, F1, per-fold details.
    """
    # Determine clips with GT
    clip_names = []
    for r in cosmos_results:
        name = Path(str(r.get("clip_path", ""))).name
        if name in gt_labels:
            clip_names.append(name)

    n_folds = len(clip_names)
    fold_details = []
    n_correct = 0

    for fold_idx, held_out_clip in enumerate(clip_names):
        # Train set: all except held-out
        train_gt = {k: v for k, v in gt_labels.items() if k != held_out_clip}

        # Find best params on train set
        random.seed(seed + fold_idx)
        best_train_acc = -1.0
        best_train_params = dict(DEFAULT_PARAMS)

        for _ in range(n_random):
            params = _random_params()
            corrector = EnhancedCorrector(params)
            corrected = corrector.correct_batch(cosmos_results, ttc_data, candidates)
            report = evaluate_enhanced(corrected, train_gt)
            if report.accuracy > best_train_acc:
                best_train_acc = report.accuracy
                best_train_params = params

        # Predict held-out clip
        corrector = EnhancedCorrector(best_train_params)
        corrected = corrector.correct_batch(cosmos_results, ttc_data, candidates)

        # Find held-out clip result
        held_out_result = None
        for r in corrected:
            if Path(str(r.get("clip_path", ""))).name == held_out_clip:
                held_out_result = r
                break

        if held_out_result is None:
            continue

        pred_sev = held_out_result.get("severity", "NONE")
        gt_sev = gt_labels[held_out_clip]
        original_sev = held_out_result.get("original_severity", pred_sev)
        reason = held_out_result.get("correction_reason")
        correct = pred_sev == gt_sev

        if correct:
            n_correct += 1

        fold_details.append({
            "clip": held_out_clip,
            "gt": gt_sev,
            "pred": pred_sev,
            "original": original_sev,
            "correct": correct,
            "train_acc": best_train_acc,
            "reason": reason,
        })

        log.info(
            "LOOCV fold %d/%d: held_out=%s gt=%s pred=%s (train_acc=%.3f)",
            fold_idx + 1, n_folds, held_out_clip, gt_sev, pred_sev, best_train_acc,
        )

    loocv_acc = n_correct / max(n_folds, 1)

    # Compute LOOCV macro-F1
    cm = [[0] * len(SEVERITY_ORDER) for _ in range(len(SEVERITY_ORDER))]
    for d in fold_details:
        if d["gt"] in _SEV_IDX and d["pred"] in _SEV_IDX:
            cm[_SEV_IDX[d["gt"]]][_SEV_IDX[d["pred"]]] += 1

    f1_scores = []
    for i in range(len(SEVERITY_ORDER)):
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        fp = sum(cm[j][i] for j in range(len(SEVERITY_ORDER))) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if sum(cm[i]) > 0:
            f1_scores.append(f1)

    loocv_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "loocv_accuracy": loocv_acc,
        "loocv_f1": loocv_f1,
        "n_folds": n_folds,
        "n_correct": n_correct,
        "details": fold_details,
    }


def load_gt_labels(gt_path: str | Path) -> dict[str, str]:
    """Load ground truth labels from CSV."""
    gt: dict[str, str] = {}
    with open(gt_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clip = Path(str(row.get("clip_path", ""))).name
            sev = str(row.get("severity", "NONE")).strip().upper()
            if clip:
                gt[clip] = sev
    return gt


def save_correction_outputs(
    corrected: list[dict],
    report: CorrectionReport,
    params: dict[str, float],
    output_dir: Path,
    loocv_report: dict | None = None,
) -> None:
    """Save all correction artifacts to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "corrected_results.json", "w", encoding="utf-8") as f:
        json.dump(corrected, f, indent=2, ensure_ascii=False)

    eval_dict = {
        "accuracy": report.accuracy,
        "macro_f1": report.macro_f1,
        "confusion_matrix": report.confusion_matrix,
        "per_class": report.per_class,
        "details": report.details,
        "n_correct": report.n_correct,
        "n_total": report.n_total,
    }
    with open(output_dir / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(eval_dict, f, indent=2, ensure_ascii=False)

    with open(output_dir / "optimal_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    if loocv_report is not None:
        with open(output_dir / "loocv_report.json", "w", encoding="utf-8") as f:
            json.dump(loocv_report, f, indent=2, ensure_ascii=False)

    log.info("Correction outputs saved to %s", output_dir)
