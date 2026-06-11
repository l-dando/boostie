"""
metrics.py
==========
Evaluation metrics for regression and classification.

All functions accept plain NumPy arrays and return a single float
(or dict for metrics that produce multiple values).

Functions
---------
  rmse          — root mean squared error (regression)
  mae           — mean absolute error (regression)
  r_squared     — coefficient of determination R² (regression)
  log_loss      — binary cross-entropy loss (binary classification)
  accuracy      — fraction of correct predictions (binary / multiclass)
  confusion_matrix  — 2×2 confusion matrix (binary)
  precision_recall  — precision and recall (binary)
"""

from __future__ import annotations
import numpy as np


# -------------------------------------------------------
# Regression metrics
# -------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

        RMSE = sqrt( mean( (y_true − y_pred)² ) )

    Lower is better.  Same units as the target.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

        MAE = mean( |y_true − y_pred| )

    Lower is better.  More robust to outliers than RMSE.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of Determination R².

        R² = 1 − SS_res / SS_tot
        SS_res = Σ (y_true − y_pred)²
        SS_tot = Σ (y_true − mean(y_true))²

    1.0 = perfect prediction; 0.0 = predicting the mean;
    negative = worse than predicting the mean.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


# -------------------------------------------------------
# Classification metrics
# -------------------------------------------------------

def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """
    Binary cross-entropy (log-loss).

        LogLoss = −mean( y·log p + (1−y)·log(1−p) )

    y_prob should be predicted probabilities in (0, 1), not raw scores.
    Lower is better; perfect predictions → 0.

    Parameters
    ----------
    y_true : binary labels {0, 1}
    y_prob : predicted probabilities
    eps    : small value to avoid log(0)
    """
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Classification accuracy.

    For binary tasks, y_pred can be raw probabilities: samples with
    predicted probability > threshold are assigned class 1.

    For multiclass tasks, pass integer class labels as y_pred.

    Parameters
    ----------
    y_true    : true labels
    y_pred    : predicted labels or probabilities
    threshold : decision boundary for binary probability inputs

    Returns
    -------
    float in [0, 1] — fraction of correct predictions
    """
    if y_pred.dtype in (np.float32, np.float64):
        y_pred = (y_pred >= threshold).astype(int)
    return float(np.mean(y_true.astype(int) == y_pred.astype(int)))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    2×2 confusion matrix for binary classification.

    Rows = actual class, Columns = predicted class.
    Order: [[TN, FP], [FN, TP]]

    Parameters
    ----------
    y_true    : binary true labels {0, 1}
    y_pred    : predicted labels or probabilities
    threshold : decision boundary if y_pred contains probabilities

    Returns
    -------
    np.ndarray of shape (2, 2), dtype int
    """
    if y_pred.dtype in (np.float32, np.float64):
        y_pred = (y_pred >= threshold).astype(int)

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return np.array([[tn, fp], [fn, tp]])


def precision_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Precision and recall for binary classification.

        Precision = TP / (TP + FP)   — of all predicted positives, how many are real?
        Recall    = TP / (TP + FN)   — of all real positives, how many did we catch?
        F1        = 2 · P · R / (P + R)

    Returns
    -------
    dict with keys 'precision', 'recall', 'f1'
    """
    cm = confusion_matrix(y_true, y_pred, threshold)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}
