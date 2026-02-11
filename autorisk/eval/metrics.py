"""Classification metrics: Accuracy and Macro-F1."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score


def compute_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Compute classification accuracy.

    Args:
        y_true: Ground truth severity labels.
        y_pred: Predicted severity labels.

    Returns:
        Accuracy as a float in [0, 1].
    """
    return float(accuracy_score(y_true, y_pred))


def compute_macro_f1(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> float:
    """Compute macro-averaged F1 score.

    Args:
        y_true: Ground truth severity labels.
        y_pred: Predicted severity labels.
        labels: Optional explicit label list for consistent ordering.

    Returns:
        Macro F1 as a float in [0, 1].
    """
    return float(f1_score(
        y_true, y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    ))
