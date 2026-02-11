"""Evaluation module: metrics, checklist, evaluator."""

from autorisk.eval.checklist import ExplanationChecklist
from autorisk.eval.evaluator import Evaluator
from autorisk.eval.metrics import compute_accuracy, compute_macro_f1

__all__ = [
    "compute_accuracy",
    "compute_macro_f1",
    "ExplanationChecklist",
    "Evaluator",
]
