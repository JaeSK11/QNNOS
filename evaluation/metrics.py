"""Evaluation metrics: accuracy, F1, classification report, confusion matrix."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred)


def compute_f1(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted"
) -> float:
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """Generate full evaluation report.

    Returns dict with accuracy, f1, classification_report string,
    and confusion_matrix.
    """
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "f1_weighted": compute_f1(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
