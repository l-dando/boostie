"""
data.py
=======
Synthetic dataset generators and data-preparation utilities.

Functions
---------
  make_regression_data   — continuous target with non-linear signal
  make_classification_data — binary 0/1 target
  make_count_data          — Poisson count target
  train_test_split         — index-based split (no sklearn dependency)
  to_numpy                 — safely convert DataFrame / Series to ndarray
"""

import numpy as np
import pandas as pd
from typing import Optional

# -------------------------------------------------------
# Dataset generators
# -------------------------------------------------------


def make_regression_data(
    n_samples: int = 500,
    n_features: int = 5,
    noise: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic regression dataset with a non-linear target.

    The true relationship uses the first three features:
        y = 2·x₀ − 1.5·x₁² + 0.5·x₂ + noise

    Parameters
    ----------
    n_samples  : number of rows
    n_features : number of columns in X (minimum 3)
    noise      : standard deviation of Gaussian noise added to y
    seed       : random seed for reproducibility

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features) — float features
    y : np.ndarray, shape (n_samples,)            — continuous target
    """
    if n_features < 3:
        raise ValueError("n_features must be at least 3 for this generator.")
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = (
        2.0 * X[:, 0]
        - 1.5 * X[:, 1] ** 2
        + 0.5 * X[:, 2]
        + rng.standard_normal(n_samples) * noise
    )
    return X, y


def make_classification_data(
    n_samples: int = 500,
    n_features: int = 5,
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic binary classification dataset.

    A log-odds signal is built from the first three features, then
    thresholded probabilistically to generate 0/1 labels.

    Parameters
    ----------
    n_samples  : number of rows
    n_features : number of columns in X (minimum 3)
    noise      : standard deviation of noise added to log-odds
    seed       : random seed for reproducibility

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,) — binary {0, 1}
    """
    if n_features < 3:
        raise ValueError("n_features must be at least 3 for this generator.")
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    log_odds = (
        1.5 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.standard_normal(n_samples) * noise
    )
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    y = (rng.uniform(size=n_samples) < prob).astype(float)
    return X, y


def make_count_data(
    n_samples: int = 500,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic Poisson count dataset.

    Log-mean is a linear function of the first two features.

    Parameters
    ----------
    n_samples  : number of rows
    n_features : number of columns in X (minimum 2)
    seed       : random seed for reproducibility

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,) — non-negative integer counts
    """
    if n_features < 2:
        raise ValueError("n_features must be at least 2 for this generator.")
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    log_mean = 1.0 + 0.8 * X[:, 0] - 0.4 * X[:, 1]
    mu = np.exp(log_mean)
    y = rng.poisson(mu).astype(float)
    return X, y


# -------------------------------------------------------
# Train / test split
# -------------------------------------------------------


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split arrays into train and test subsets.

    Parameters
    ----------
    X         : feature matrix, shape (n, p)
    y         : target vector,  shape (n,)
    test_size : fraction of samples to use as the test set (0 < test_size < 1)
    shuffle   : whether to shuffle before splitting
    seed      : random seed used when shuffle=True

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    n = len(y)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    split = int(n * (1.0 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# -------------------------------------------------------
# Conversion helpers
# -------------------------------------------------------


def to_numpy(
    obj,
    dtype: Optional[type] = None,
) -> np.ndarray:
    """
    Convert a pandas DataFrame, Series, or list to a NumPy array.

    Parameters
    ----------
    obj   : array-like — can be np.ndarray, pd.DataFrame,
            pd.Series, or a plain Python list / nested list
    dtype : optional NumPy dtype to cast to

    Returns
    -------
    np.ndarray
    """
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        arr = obj.to_numpy()
    else:
        arr = np.asarray(obj)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr
