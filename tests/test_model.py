import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from boostie.model import boostieModel


_n_rows = st.integers(min_value=2, max_value=20)
_n_cols = st.integers(min_value=1, max_value=5)
_feature_vals = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
_target_vals = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
_raw_vals = st.floats(min_value=-8.0, max_value=8.0, allow_nan=False, allow_infinity=False)


@st.composite
def regression_dataset(draw):
    n = draw(_n_rows)
    m = draw(_n_cols)
    X = draw(arrays(np.float64, (n, m), elements=_feature_vals))
    y = draw(arrays(np.float64, n, elements=_target_vals))
    return X, y


@st.composite
def binary_dataset(draw):
    n = draw(_n_rows)
    m = draw(_n_cols)
    X = draw(arrays(np.float64, (n, m), elements=_feature_vals))
    y = draw(arrays(np.float64, n, elements=st.sampled_from([0.0, 1.0])))
    return X, y


def test_set_tweedie_power_rejects_out_of_range():
    model = boostieModel(objective="tweedie")
    with pytest.raises(ValueError, match="range"):
        model.set_tweedie_power(1.0)
    with pytest.raises(ValueError, match="range"):
        model.set_tweedie_power(2.0)


@given(p=st.floats(min_value=1.0001, max_value=1.9999, allow_nan=False, allow_infinity=False))
def test_set_tweedie_power_accepts_valid_values(p):
    model = boostieModel(objective="tweedie")
    model.set_tweedie_power(p)
    assert model.tweedie_power == p


def test_predict_raw_raises_when_unfitted():
    model = boostieModel()
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict_raw(np.zeros((2, 1)))


def test_feature_importances_raises_when_unfitted():
    model = boostieModel()
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.feature_importances(2)


@given(data=regression_dataset(), n_estimators=st.integers(min_value=1, max_value=5))
def test_fit_predict_regression_identity_link(data, n_estimators):
    X, y = data
    model = boostieModel(
        objective="regression",
        n_estimators=n_estimators,
        max_depth=2,
        learning_rate=0.2,
    )
    model.fit(X, y)

    raw = model.predict_raw(X)
    pred = model.predict(X)

    assert model.n_trees == n_estimators
    assert raw.shape == (X.shape[0],)
    assert pred.shape == (X.shape[0],)
    np.testing.assert_allclose(pred, raw)
    assert "status=fitted" in repr(model)


@given(data=binary_dataset(), n_estimators=st.integers(min_value=1, max_value=5))
def test_predict_proba_binary_properties(data, n_estimators):
    X, y = data
    model = boostieModel(
        objective="binary",
        n_estimators=n_estimators,
        max_depth=2,
        learning_rate=0.2,
    )
    model.fit(X, y)

    p1 = model.predict(X)
    proba = model.predict_proba(X)

    assert proba.shape == (X.shape[0], 2)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(proba[:, 1], p1, atol=1e-9)


def test_predict_proba_raises_for_non_binary_objective():
    model = boostieModel(objective="regression")
    with pytest.raises(ValueError, match="objective='binary'"):
        model.predict_proba(np.zeros((3, 2)))


@given(raw=arrays(np.float64, 20, elements=_raw_vals))
def test_apply_link_behaviour(raw):
    regression = boostieModel(objective="regression")
    binary = boostieModel(objective="binary")
    classification = boostieModel(objective="classification")
    poisson = boostieModel(objective="poisson")
    tweedie = boostieModel(objective="tweedie")

    np.testing.assert_allclose(regression._apply_link(raw), raw)

    binary_vals = binary._apply_link(raw)
    classification_vals = classification._apply_link(raw)
    assert np.all(binary_vals >= 0.0)
    assert np.all(binary_vals <= 1.0)
    np.testing.assert_allclose(binary_vals, classification_vals)

    np.testing.assert_allclose(poisson._apply_link(raw), np.exp(raw))
    np.testing.assert_allclose(tweedie._apply_link(raw), np.exp(raw))


@given(data=regression_dataset())
def test_feature_importances_shape_and_sum(data):
    X, y = data
    n_features = X.shape[1]
    model = boostieModel(objective="regression", n_estimators=4, max_depth=2, learning_rate=0.1)
    model.fit(X, y)
    importances = model.feature_importances(n_features)

    assert importances.shape == (n_features,)
    assert np.all(importances >= 0.0)
    total = importances.sum()
    assert np.isclose(total, 0.0) or np.isclose(total, 1.0)


def test_preprocess_return_df_false_with_label_encoding():
    model = boostieModel()
    X = pd.DataFrame({"city": ["a", "b", "a"], "value": [1, 2, 3]})
    vals, cols = model.preprocess(
        X,
        feature={"city": "label_encoder"},
        inplace=False,
        return_df=False,
    )

    assert isinstance(vals, np.ndarray)
    assert vals.shape == (3, 3)
    assert cols == ["city", "value", "city_label_encoded"]


def test_preprocess_inplace_replaces_column_for_one_hot():
    model = boostieModel()
    X = pd.DataFrame({"city": ["a", "b", "a"], "value": [1, 2, 3]})
    out = model.preprocess(
        X,
        feature={"city": "one_hot_encoding"},
        inplace=True,
        return_df=True,
    )

    assert isinstance(out, pd.DataFrame)
    assert "city" not in out.columns
    assert "city_a" in out.columns
    assert "city_b" in out.columns
    assert "value" in out.columns


def test_fit_with_early_stopping_stops_before_n_estimators():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 1.0, 1.0, 1.0])
    model = boostieModel(
        objective="regression",
        n_estimators=10,
        learning_rate=0.0,
        early_stopping_rounds=1,
    )
    model.fit(X, y)
    assert model.n_trees < 10
