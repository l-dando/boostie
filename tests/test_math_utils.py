import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from boostie.math_utils import leaf_score, optimal_weight, split_gain

# ---------------------------------------------------------------------------
# Reusable strategies
# ---------------------------------------------------------------------------
# Positive floats for quantities that must be > 0 (hessian sums, regularisation)
_pos = st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False)
# Arbitrary floats for gradient sums
_any = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
# Smaller range to avoid catastrophic cancellation in split_gain symmetry test
_g = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
_h = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)
_lam = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)
_gam = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Fixed-value regression tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Property-based tests — leaf_score
# ---------------------------------------------------------------------------


@given(g_sum=_any, h_sum=_pos, reg_lambda=_pos)
def test_leaf_score_non_negative(g_sum, h_sum, reg_lambda):
    """Score = g² / (h + λ) ≥ 0 for all valid inputs."""
    assert leaf_score(g_sum, h_sum, reg_lambda) >= 0.0


@given(g_sum=_any, h_sum=_pos, reg_lambda=_pos)
def test_leaf_score_symmetric_in_g(g_sum, h_sum, reg_lambda):
    """Negating g leaves the score unchanged because g is squared."""
    assert np.isclose(
        leaf_score(g_sum, h_sum, reg_lambda),
        leaf_score(-g_sum, h_sum, reg_lambda),
    )


@given(g_sum=_any, h_sum=_pos, reg_lambda_lo=_pos, reg_lambda_hi=_pos)
def test_leaf_score_decreases_with_lambda(g_sum, h_sum, reg_lambda_lo, reg_lambda_hi):
    """Larger regularisation ⇒ smaller (or equal) structural score."""
    lo, hi = min(reg_lambda_lo, reg_lambda_hi), max(reg_lambda_lo, reg_lambda_hi)
    assert leaf_score(g_sum, h_sum, lo) >= leaf_score(g_sum, h_sum, hi)


# ---------------------------------------------------------------------------
# Property-based tests — optimal_weight
# ---------------------------------------------------------------------------


@given(h_sum=_pos, reg_lambda=_pos)
def test_optimal_weight_zero_grad_gives_zero(h_sum, reg_lambda):
    """No gradient signal ⇒ zero leaf weight."""
    assert optimal_weight(0.0, h_sum, reg_lambda) == 0.0


@given(g_sum=_any, h_sum=_pos, reg_lambda=_pos)
def test_optimal_weight_antisymmetric_in_g(g_sum, h_sum, reg_lambda):
    """Flipping the sign of g flips the weight: w*(-g) = -w*(g)."""
    assert np.isclose(
        optimal_weight(-g_sum, h_sum, reg_lambda),
        -optimal_weight(g_sum, h_sum, reg_lambda),
    )


@given(g_sum=_any, h_sum=_pos, reg_lambda=_pos)
def test_leaf_score_equals_g_times_neg_optimal_weight(g_sum, h_sum, reg_lambda):
    """Mathematical identity: Score(j) = g_sum × (−w*_j)."""
    score = leaf_score(g_sum, h_sum, reg_lambda)
    weight = optimal_weight(g_sum, h_sum, reg_lambda)
    assert np.isclose(score, g_sum * (-weight), rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Property-based tests — split_gain
# ---------------------------------------------------------------------------


@given(g_left=_g, h_left=_h, g_right=_g, h_right=_h, reg_lambda=_lam, reg_gamma=_gam)
def test_split_gain_symmetric_left_right(g_left, h_left, g_right, h_right, reg_lambda, reg_gamma):
    """Swapping left and right children does not change the gain."""
    g_parent = g_left + g_right
    h_parent = h_left + h_right
    gain_lr = split_gain(
        g_left, h_left, g_right, h_right, g_parent, h_parent, reg_lambda, reg_gamma
    )
    gain_rl = split_gain(
        g_right, h_right, g_left, h_left, g_parent, h_parent, reg_lambda, reg_gamma
    )
    assert np.isclose(gain_lr, gain_rl, rtol=1e-9, atol=1e-9)


@given(g_parent=_g, h_parent=_h, reg_lambda=_lam, reg_gamma=_gam)
def test_split_gain_trivial_split_equals_neg_gamma(g_parent, h_parent, reg_lambda, reg_gamma):
    """Sending all samples to one child scores zero improvement minus the complexity penalty.

    score_left = score_parent, score_right = 0  ⇒  gain = −γ.
    """
    gain = split_gain(
        g_parent, h_parent,  # left child = parent
        0.0, 0.0,            # right child is empty (g=0, h=0)
        g_parent, h_parent,
        reg_lambda, reg_gamma,
    )
    assert np.isclose(gain, -reg_gamma, rtol=1e-9, atol=1e-9)
