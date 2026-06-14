import numpy as np
import pandas as pd

def one_hot_encoding(X: np.array, col_name: str) -> tuple[np.ndarray, list[str]]:
    """
    One-hot encode categorical features in the dataset.

    Parameters
    ----------
    X : feature matrix, shape (n_samples, n_features) or pd.Series
    col_name : name of the categorical column to encode

    Returns
    -------
    X_encoded : one-hot encoded feature matrix, shape (n_samples, n_encoded_features)
    """
    categories, inverse = np.unique(X, return_inverse=True)
    one_hot = np.zeros((X.size, categories.size))
    one_hot[np.arange(X.size), inverse] = 1

    unique_vals = []
    for item in X:
        if item not in unique_vals:
            unique_vals.append(item)
    cols = []
    for val in unique_vals:
        cols.append(f"{col_name}_{val}")

    return one_hot, cols


# -------------------------------------------------------
# Registry
# -------------------------------------------------------
# Maps objective name → gradient function.
# Extend this dict to add new objectives.

PREPROCESSORS: dict[str, callable] = {
    "one_hot_encoding": one_hot_encoding,
}


def get_preprocesser(name: str) -> callable:
    """
    Look up a preprocessing function by technique name.

    Parameters
    ----------
    name : one of the keys in PREPROCESSORS

    Returns
    -------
    A callable PREPROCESSOR FUNCTION

    Raises
    ------
    ValueError if the name is not registered.
    """
    if name not in PREPROCESSORS:
        raise ValueError(
            f"Unknown preprocessor '{name}'. " f"Available: {list(PREPROCESSORS.keys())}"
        )
    return PREPROCESSORS[name]
