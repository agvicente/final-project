"""Detection metrics for anomaly detection evaluation.

Provides confusion-matrix-based metrics (TP, FP, FN, TN, FPR, Recall,
Precision, F1) and anomaly rate computation.
"""

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute detection metrics from ground truth and predictions.

    Args:
        y_true: Ground truth labels (0 = normal, 1 = anomaly).
        y_pred: Predicted labels (0 = normal, 1 = anomaly).

    Returns:
        Dictionary with keys: TP, FP, FN, TN, FPR, Recall, Precision, F1,
        anomaly_rate.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))

    # FPR = FP / (FP + TN)
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    # Recall (TPR) = TP / (TP + FN)
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # Precision = TP / (TP + FP)
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # F1 = 2 * Precision * Recall / (Precision + Recall)
    F1 = (
        2 * Precision * Recall / (Precision + Recall)
        if (Precision + Recall) > 0
        else 0.0
    )

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "FPR": FPR,
        "Recall": Recall,
        "Precision": Precision,
        "F1": F1,
        "anomaly_rate": anomaly_rate(y_pred),
    }


def anomaly_rate(y_pred: np.ndarray) -> float:
    """Compute the fraction of predictions that are anomalies.

    Args:
        y_pred: Predicted labels (0 = normal, 1 = anomaly).

    Returns:
        Fraction of predictions labeled as anomaly.
    """
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_pred) == 0:
        return 0.0
    return float(np.mean(y_pred))
