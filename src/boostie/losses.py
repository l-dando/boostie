"""
losses.py
=========
Loss functions and their first- and second-order gradients (g, h).

XGBoost's tree-growing algorithm does not need the loss value itself —
only the per-sample gradient (g = dL/dŷ) and hessian (h = d²L/dŷ²).
These are computed at the start of every boosting round from the
*current* predictions ŷ and the true labels y.

Adding a new objective
----------------------
1. Write a function with signature:
       def my_loss_gradients(y: np.ndarray, y_pred: np.ndarray)
                             -> tuple[np.ndarray, np.ndarray]
   where y_pred is the *raw model margin* (before any link function).
2. Register it in the OBJECTIVES dict at the bottom of this file.
3. Pass its key as `objective=` when constructing XGBoostModel.

Supported objectives
--------------------
  'regression'  — squared-error (MSE), for continuous targets
  'binary'      — binary cross-entropy (log-loss), for 0/1 targets
  'poisson'     — Poisson log-likelihood, for count targets (y ≥ 0)
  'tweedie'     — Tweedie log-likelihood, for non-negative targets
"""

import numpy as np

# -------------------------------------------------------
# Squared-error (regression)
# -------------------------------------------------------
# Loss:  L(y, ŷ) = ½ (y − ŷ)²
# g    = ŷ − y          (the plain residual)
# h    = 1              (constant hessian)
# -------------------------------------------------------


def squared_error_gradients(
    y: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    First and second derivatives of the squared-error loss.

    Parameters
    ----------
    y      : true target values, shape (n,)
    y_pred : current raw predictions, shape (n,)

    Returns
    -------
    g : first-order gradients, shape (n,)
    h : second-order gradients (hessians), shape (n,)
    """
    grad = y_pred - y
    hess = np.ones_like(y, dtype=float)
    return grad, hess


# -------------------------------------------------------
# Binary cross-entropy (logistic / binary classification)
# -------------------------------------------------------
# y_pred is the raw log-odds (margin), NOT a probability.
# Link function: p = sigmoid(y_pred)
#
# Loss:  L(y, p) = −[y log p + (1−y) log(1−p)]
# g    = p − y
# h    = p (1 − p)      (variance of Bernoulli)
# -------------------------------------------------------


def log_loss_gradients(
    y: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    First and second derivatives of the binary cross-entropy loss.

    Parameters
    ----------
    y      : binary labels {0, 1}, shape (n,)
    y_pred : current raw log-odds predictions, shape (n,)

    Returns
    -------
    g : first-order gradients, shape (n,)
    h : second-order gradients (hessians), shape (n,)
    """
    prob = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid
    grad = prob - y
    hess = prob * (1.0 - prob)
    # Clip hessian away from zero to avoid division issues in leaves
    hess = np.clip(hess, 1e-6, None)
    return grad, hess


# -------------------------------------------------------
# Poisson regression (count targets)
# -------------------------------------------------------
# y_pred is the raw log-mean (margin), NOT the predicted count.
# Link function: μ = exp(y_pred)
#
# Loss:  L(y, μ) = μ − y log μ     (negative log-likelihood)
# g    = exp(y_pred) − y   =  μ − y
# h    = exp(y_pred)        =  μ
# -------------------------------------------------------


def poisson_gradients(
    y: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    First and second derivatives of the Poisson log-likelihood loss.

    Parameters
    ----------
    y      : non-negative count targets, shape (n,)
    y_pred : current raw log-mean predictions, shape (n,)

    Returns
    -------
    g : first-order gradients, shape (n,)
    h : second-order gradients (hessians), shape (n,)
    """
    mu = np.exp(y_pred)
    grad = mu - y
    hess = mu
    return grad, hess


# -------------------------------------------------------
# Tweedie regression (non-negative targets)
# -------------------------------------------------------
# y_pred is the raw prediction (margin), NOT the predicted value.
# Link function: μ = exp(y_pred)
#
# Loss:  L(y, μ) = ((y * exp((1 - p) * y_pred))/(1 - p)) + ((exp((2 - p) * y_pred))/(2 - p))   (negative log-likelihood)
# g    = -y * exp((1 - p) * y_pred) + exp((1 - p) * y_pred)
# h    = -y * (1 - p) * exp((1 - p) * y_pred) + (1 - p) * exp((1 - p) * y_pred) 
# -------------------------------------------------------


def tweedie(
    y: np.ndarray,
    y_pred: np.ndarray,
    p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    First and second derivatives of the Tweedie loss.

    Parameters
    ----------
    y      : target values, shape (n,)
    y_pred : current raw predictions, shape (n,)
    p      : Tweedie power parameter

    Returns
    -------
    g : first-order gradients, shape (n,)
    h : second-order gradients (hessians), shape (n,)
    """
    y_pred = np.clip(y_pred, -20, 20)
    y_safe = np.maximum(y, 1e-6)  # Avoid log(0) for y=0

    grad = -y_safe * np.exp((1 - p) * y_pred) + np.exp((2 - p) * y_pred)
    hess = -y_safe * (1 - p) * np.exp((1 - p) * y_pred) + (2 - p) * np.exp((2 - p) * y_pred)
    return grad, hess


# -------------------------------------------------------
# Registry
# -------------------------------------------------------
# Maps objective name → gradient function.
# Extend this dict to add new objectives.

OBJECTIVES: dict[str, callable] = {
    "regression": squared_error_gradients,
    "binary": log_loss_gradients,
    "classification": log_loss_gradients,  # alias for binary
    "poisson": poisson_gradients,
    "tweedie": tweedie,
}


def get_objective(name: str) -> callable:
    """
    Look up a gradient function by objective name.

    Parameters
    ----------
    name : one of the keys in OBJECTIVES

    Returns
    -------
    A callable (y, y_pred) -> (g, h)

    Raises
    ------
    ValueError if the name is not registered.
    """
    if name not in OBJECTIVES:
        raise ValueError(
            f"Unknown objective '{name}'. " f"Available: {list(OBJECTIVES.keys())}"
        )
    return OBJECTIVES[name]
