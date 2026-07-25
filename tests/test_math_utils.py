import numpy as np

from boostie.math_utils import leaf_score, optimal_weight, split_gain


def test_leaf_score_matches_formula():
    g_sum, h_sum, reg_lambda = 3.0, 2.0, 1.0
    assert np.isclose(leaf_score(g_sum, h_sum, reg_lambda), (g_sum**2) / (h_sum + reg_lambda))


def test_optimal_weight_matches_formula():
    g_sum, h_sum, reg_lambda = 4.0, 3.0, 2.0
    assert optimal_weight(g_sum, h_sum, reg_lambda) == -g_sum / (h_sum + reg_lambda)


def test_split_gain_matches_manual_computation():
    g_left, h_left = 2.0, 1.5
    g_right, h_right = -1.0, 1.0
    g_parent, h_parent = g_left + g_right, h_left + h_right
    reg_lambda, reg_gamma = 1.0, 0.1

    gain = split_gain(
        g_left,
        h_left,
        g_right,
        h_right,
        g_parent,
        h_parent,
        reg_lambda,
        reg_gamma,
    )

    expected = (
        0.5
        * (
            (g_left**2) / (h_left + reg_lambda)
            + (g_right**2) / (h_right + reg_lambda)
            - (g_parent**2) / (h_parent + reg_lambda)
        )
        - reg_gamma
    )
    assert np.isclose(gain, expected)
