import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from boostie.losses import get_objective, log_loss_gradients, squared_error_gradients

# ---------------------------------------------------------------------------
# Reusable strategies
# ---------------------------------------------------------------------------
_n = st.integers(min_value=1, max_value=100)
_scalar_floats = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
_logit_floats = st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False)
_binary_labels = st.sampled_from([0.0, 1.0])


def _y_array(n):
    return arrays(np.float64, n, elements=_scalar_floats)


def _y_pred_array(n):
    return arrays(np.float64, n, elements=_scalar_floats)


def _logit_array(n):
    return arrays(np.float64, n, elements=_logit_floats)


def _binary_array(n):
    return arrays(np.float64, n, elements=_binary_labels)


# ---------------------------------------------------------------------------
# Fixed-value regression tests
# ---------------------------------------------------------------------------


def test_squared_error_gradients():
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 1.5, 2.5])

    grad, hess = squared_error_gradients(y, y_pred)

    np.testing.assert_allclose(grad, np.array([0.5, -0.5, -0.5]))
    np.testing.assert_allclose(hess, np.ones_like(y))


def test_log_loss_gradients_hessian_is_positive():
    y = np.array([0.0, 1.0, 1.0, 0.0])
    y_pred = np.array([-5.0, 0.0, 5.0, 1.0])

    grad, hess = log_loss_gradients(y, y_pred)

    assert grad.shape == y.shape
    assert hess.shape == y.shape
    assert np.all(hess > 0)


def test_get_objective_raises_for_unknown_objective():
    with pytest.raises(ValueError, match="Unknown objective"):
        get_objective("not-a-real-objective")


# ---------------------------------------------------------------------------
# Property-based tests — squared_error_gradients
# ---------------------------------------------------------------------------


@given(n=_n, data=st.data())
def test_squared_error_output_shapes(n, data):
    """Gradient and hessian shapes always match input shape."""
    y = data.draw(_y_array(n))
    y_pred = data.draw(_y_pred_array(n))
    grad, hess = squared_error_gradients(y, y_pred)
    assert grad.shape == (n,)
    assert hess.shape == (n,)


@given(n=_n, data=st.data())
def test_squared_error_perfect_prediction_zero_grad(n, data):
    """When predictions equal targets, gradient is everywhere zero."""
    y = data.draw(_y_array(n))
    grad, _ = squared_error_gradients(y, y)
    np.testing.assert_allclose(grad, 0.0)


@given(n=_n, data=st.data())
def test_squared_error_hessian_always_ones(n, data):
    """Hessian of squared-error loss is identically 1 for all inputs."""
    y = data.draw(_y_array(n))
    y_pred = data.draw(_y_pred_array(n))
    _, hess = squared_error_gradients(y, y_pred)
    np.testing.assert_allclose(hess, 1.0)


@given(n=_n, data=st.data())
def test_squared_error_gradient_sign_when_over_predicting(n, data):
    """grad = ŷ − y: over-predictions (ŷ > y) give positive gradients."""
    y = data.draw(arrays(np.float64, n, elements=st.floats(0.0, 49.0, allow_nan=False)))
    y_pred = data.draw(arrays(np.float64, n, elements=st.floats(50.0, 100.0, allow_nan=False)))
    grad, _ = squared_error_gradients(y, y_pred)
    assert np.all(grad > 0)


@given(n=_n, data=st.data())
def test_squared_error_gradient_antisymmetric(n, data):
    """Swapping y and y_pred negates the gradient: grad(y, ŷ) = −grad(ŷ, y)."""
    y = data.draw(_y_array(n))
    y_pred = data.draw(_y_pred_array(n))
    grad_fwd, _ = squared_error_gradients(y, y_pred)
    grad_rev, _ = squared_error_gradients(y_pred, y)
    np.testing.assert_allclose(grad_fwd, -grad_rev)


# ---------------------------------------------------------------------------
# Property-based tests — log_loss_gradients
# ---------------------------------------------------------------------------


@given(n=_n, data=st.data())
def test_log_loss_output_shapes(n, data):
    """Gradient and hessian shapes always match input shape."""
    y = data.draw(_binary_array(n))
    y_pred = data.draw(_logit_array(n))
    grad, hess = log_loss_gradients(y, y_pred)
    assert grad.shape == (n,)
    assert hess.shape == (n,)


@given(n=_n, data=st.data())
def test_log_loss_hessian_always_positive(n, data):
    """h = p(1−p) clipped at 1e-6 is always strictly positive."""
    y = data.draw(_binary_array(n))
    y_pred = data.draw(_logit_array(n))
    _, hess = log_loss_gradients(y, y_pred)
    assert np.all(hess > 0)


@given(n=_n, data=st.data())
def test_log_loss_hessian_bounded(n, data):
    """h = p(1−p) ≤ 0.25 (maximum at p=0.5) and ≥ 1e-6 (clip floor)."""
    y = data.draw(_binary_array(n))
    y_pred = data.draw(_logit_array(n))
    _, hess = log_loss_gradients(y, y_pred)
    assert np.all(hess >= 1e-6)
    assert np.all(hess <= 0.25 + 1e-9)


@given(n=_n, data=st.data())
def test_log_loss_y0_gradient_positive(n, data):
    """When y=0, grad = prob − 0 = prob > 0 always."""
    y = np.zeros(n)
    y_pred = data.draw(_logit_array(n))
    grad, _ = log_loss_gradients(y, y_pred)
    assert np.all(grad > 0)


@given(n=_n, data=st.data())
def test_log_loss_y1_gradient_negative(n, data):
    """When y=1, grad = prob − 1 < 0 always (since prob < 1)."""
    y = np.ones(n)
    y_pred = data.draw(_logit_array(n))
    grad, _ = log_loss_gradients(y, y_pred)
    assert np.all(grad < 0)
