from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_directional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | tuple[str, ...] = ("up", "down"),
) -> dict[str, float]:
    labels = list(labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }
