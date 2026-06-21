import numpy as np
import pandas as pd

def one_hot_encoding(X: np.array, col_name: str, dropna: bool = False) -> tuple[np.ndarray, list[str]]:
    """
    One-hot encode categorical features in the dataset.

    Parameters
    ----------
    X : feature matrix, shape (n_samples, n_features) or pd.Series
    col_name : name of the categorical column to encode
    dropna : whether to drop the column for missing values

    Returns
    -------
    X_encoded : one-hot encoded feature matrix, shape (n_samples, n_encoded_features)
    """
    values = np.asarray(X, dtype=object)
    series = pd.Series(values)

    # Keep first-seen order and avoid mixed-type sorting errors.
    filled = series.where(series.notna(), "_missing")
    codes, categories = pd.factorize(filled, sort=False)

    one_hot = np.zeros((values.size, categories.size), dtype=float)
    one_hot[np.arange(values.size), codes] = 1.0

    cols = [f"{col_name}_{val}" for val in categories]

    if dropna:
        # Drop the column for missing values if requested.
        missing_col = f"{col_name}__missing"
        if missing_col in cols:
            idx = cols.index(missing_col)
            one_hot = np.delete(one_hot, idx, axis=1)
            cols.pop(idx)

    return one_hot, cols


def label_encoder(X: np.array, col_name: str, **kwargs) -> tuple[np.ndarray, list[str]]:
    """
    Label encode categorical features in the dataset.

    Parameters
    ----------
    X : feature matrix, shape (n_samples, n_features) or pd.Series
    col_name : name of the categorical column to encode

    Returns
    -------
    X_encoded : label encoded feature matrix, shape (n_samples, 1)
    """
    series = pd.Series(np.asarray(X, dtype=object))
    filled = series.where(series.notna(), "_missing")
    codes, _ = pd.factorize(filled, sort=False)
    encoded = codes.reshape(-1, 1)

    return encoded, [f"{col_name}_label_encoded"]
    return encoded.T, [f"{col_name}_label_encoded"]


# -------------------------------------------------------
# Registry
# -------------------------------------------------------
# Maps objective name → gradient function.
# Extend this dict to add new objectives.

PREPROCESSORS: dict[str, callable] = {
    "one_hot_encoding": one_hot_encoding,
    "label_encoder": label_encoder,
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
