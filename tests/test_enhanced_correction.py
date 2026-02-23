"""Tests for enhanced signal-based severity correction."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from autorisk.eval.enhanced_correction import (
    DEFAULT_PARAMS,
    CorrectionReport,
    EnhancedCorrector,
    evaluate_enhanced,
    load_gt_labels,
    save_correction_outputs,
)


@pytest.fixture()
def corrector():
    return EnhancedCorrector(DEFAULT_PARAMS)


# --- Individual rule tests ---

class TestR1NoneDetection:
    def test_triggers_when_safe(self, corrector):
        sev, reason = corrector.correct_single(
            severity="LOW", fused_score=0.40, min_ttc=1.2, n_critical=0,
        )
        assert sev == "NONE"
        assert reason is not None and "R1" in reason

    def test_skips_when_ttc_low(self, corrector):
        sev, reason = corrector.correct_single(
            severity="LOW", fused_score=0.40, min_ttc=0.5, n_critical=0,
        )
        assert sev != "NONE" or reason is None  # R1 should not trigger


class TestR2HighDetection:
    def test_triggers_when_dangerous(self, corrector):
        sev, reason = corrector.correct_single(
            severity="LOW", fused_score=0.50, min_ttc=0.20, n_critical=5,
        )
        assert sev == "HIGH"
        assert reason is not None and "R2" in reason

    def test_skips_when_few_critical(self, corrector):
        sev, reason = corrector.correct_single(
            severity="LOW", fused_score=0.50, min_ttc=0.20, n_critical=2,
        )
        assert sev != "HIGH" or (reason is not None and "R2" not in reason)


class TestR3MediumPromotion:
    def test_promotes_low_to_medium(self, corrector):
        sev, reason = corrector.correct_single(
            severity="LOW", fused_score=0.70, min_ttc=0.5, n_critical=2,
        )
        assert sev == "MEDIUM"
        assert reason is not None and "R3" in reason

    def test_skips_non_low(self, corrector):
        sev, reason = corrector.correct_single(
            severity="MEDIUM", fused_score=0.70, min_ttc=0.5, n_critical=2,
        )
        # R3 only applies to LOW severity
        assert reason is None or "R3" not in str(reason)


class TestR4HighDemotion:
    def test_demotes_high_to_medium(self, corrector):
        sev, reason = corrector.correct_single(
            severity="HIGH", fused_score=0.60, min_ttc=1.8, n_critical=1,
        )
        assert sev == "MEDIUM"
        assert reason is not None and "R4" in reason

    def test_skips_non_high(self, corrector):
        sev, reason = corrector.correct_single(
            severity="MEDIUM", fused_score=0.60, min_ttc=1.8, n_critical=1,
        )
        assert reason is None or "R4" not in str(reason)


class TestR5MediumDemotion:
    def test_demotes_medium_to_low(self, corrector):
        sev, reason = corrector.correct_single(
            severity="MEDIUM", fused_score=0.40, min_ttc=0.7, n_critical=2,
        )
        assert sev == "LOW"
        assert reason is not None and "R5" in reason


class TestR6NoneUpgrade:
    def test_upgrades_none_to_low(self, corrector):
        sev, reason = corrector.correct_single(
            severity="NONE", fused_score=0.60, min_ttc=0.4, n_critical=3,
        )
        assert sev == "LOW"
        assert reason is not None and "R6" in reason

    def test_skips_when_ttc_safe(self, corrector):
        sev, reason = corrector.correct_single(
            severity="NONE", fused_score=0.60, min_ttc=0.8, n_critical=3,
        )
        # TTC 0.8 > ttc_none_up=0.5, so R6 should not trigger
        assert sev == "NONE"


# --- Confidence gate ---

class TestConfidenceGate:
    def test_high_confidence_skips_correction(self):
        corrector = EnhancedCorrector({"conf_gate": 0.5, **DEFAULT_PARAMS})
        corrector.params["conf_gate"] = 0.5
        sev, reason = corrector.correct_single(
            severity="LOW", fused_score=0.40, min_ttc=1.5, n_critical=0,
            confidence=0.8,
        )
        # Confidence 0.8 > gate 0.5 => skip correction
        assert sev == "LOW"
        assert reason is None


# --- Batch correction ---

class TestBatchCorrection:
    def test_corrects_batch(self, corrector):
        cosmos_results = [
            {"clip_path": "clips/a.mp4", "severity": "LOW", "fused_score": 0.40, "confidence": 0.0},
            {"clip_path": "clips/b.mp4", "severity": "NONE", "fused_score": 0.60, "confidence": 0.0},
        ]
        ttc_data = [
            {"clip_path": "clips/a.mp4", "min_ttc": 1.5, "n_critical": 0},
            {"clip_path": "clips/b.mp4", "min_ttc": 0.3, "n_critical": 5},
        ]
        corrected = corrector.correct_batch(cosmos_results, ttc_data)
        assert len(corrected) == 2
        assert all("original_severity" in r for r in corrected)
        assert all("correction_reason" in r for r in corrected)

    def test_original_severity_preserved(self, corrector):
        cosmos_results = [
            {"clip_path": "clips/a.mp4", "severity": "LOW", "fused_score": 0.40, "confidence": 0.0},
        ]
        ttc_data = [
            {"clip_path": "clips/a.mp4", "min_ttc": 1.5, "n_critical": 0},
        ]
        corrected = corrector.correct_batch(cosmos_results, ttc_data)
        assert corrected[0]["original_severity"] == "LOW"


# --- Evaluation ---

class TestEvaluateEnhanced:
    def test_perfect_accuracy(self):
        results = [
            {"clip_path": "clips/a.mp4", "severity": "HIGH", "original_severity": "HIGH"},
            {"clip_path": "clips/b.mp4", "severity": "LOW", "original_severity": "LOW"},
        ]
        gt = {"a.mp4": "HIGH", "b.mp4": "LOW"}
        report = evaluate_enhanced(results, gt)
        assert report.accuracy == 1.0
        assert report.n_correct == 2
        assert report.n_total == 2

    def test_zero_accuracy(self):
        results = [
            {"clip_path": "clips/a.mp4", "severity": "NONE", "original_severity": "NONE"},
        ]
        gt = {"a.mp4": "HIGH"}
        report = evaluate_enhanced(results, gt)
        assert report.accuracy == 0.0


# --- Save/Load ---

class TestSaveLoad:
    def test_save_outputs(self, corrector, tmp_path):
        report = CorrectionReport(
            accuracy=0.5, macro_f1=0.4, n_correct=1, n_total=2,
            confusion_matrix=[[1, 0], [1, 0]],
            per_class={"NONE": {"count": 1, "correct": 1, "recall": 1.0}},
            details=[],
        )
        save_correction_outputs([], report, DEFAULT_PARAMS, tmp_path)
        assert (tmp_path / "eval_report.json").exists()
        assert (tmp_path / "optimal_params.json").exists()
        assert (tmp_path / "corrected_results.json").exists()

    def test_load_gt_labels(self, tmp_path):
        gt_csv = tmp_path / "gt.csv"
        gt_csv.write_text("clip_path,severity\nclips/a.mp4,HIGH\nclips/b.mp4,LOW\n")
        gt = load_gt_labels(gt_csv)
        assert gt["a.mp4"] == "HIGH"
        assert gt["b.mp4"] == "LOW"
