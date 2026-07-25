import numpy as np
import pytest

from boostie.losses import get_objective, log_loss_gradients, squared_error_gradients


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
