"""
math_utils.py
=============
Core mathematical formulas derived from XGBoost's second-order
Taylor expansion of the objective function.

Background
----------
At each boosting round, XGBoost approximates the loss as:

    Obj ≈ Σᵢ [ gᵢ fₜ(xᵢ) + ½ hᵢ fₜ(xᵢ)² ]  +  Ω(fₜ)

where Ω(fₜ) = γ T + ½ λ Σⱼ wⱼ²
      T  = number of leaves
      wⱼ = prediction (weight) of leaf j
      λ  = L2 regularisation strength

For a fixed tree structure, the optimal weight of leaf j is:

    w*ⱼ = − (Σᵢ∈Iⱼ gᵢ) / (Σᵢ∈Iⱼ hᵢ + λ)

and the resulting (negated) objective value for that leaf is:

    Score(j) = (Σᵢ∈Iⱼ gᵢ)² / (Σᵢ∈Iⱼ hᵢ + λ)

The gain from splitting node S into left child L and right child R:

    Gain = ½ [ Score(L) + Score(R) − Score(S) ] − γ

A split is only accepted when Gain > 0.

These three formulas — leaf_score, optimal_weight, split_gain —
are all that the tree-growing algorithm needs from the maths layer.
"""

from __future__ import annotations
import numpy as np


def leaf_score(g_sum: float, h_sum: float, reg_lambda: float) -> float:
    """
    Structural score of a leaf node (higher = better fit).

    This is the value of the reduced objective for a single leaf
    after substituting the optimal weight w*.

    Formula:  Score = (Σg)² / (Σh + λ)

    Parameters
    ----------
    g_sum      : sum of first-order gradients for samples in this leaf
    h_sum      : sum of second-order gradients (hessians) for this leaf
    reg_lambda : L2 regularisation coefficient λ

    Returns
    -------
    float — the structural score (always ≥ 0)
    """
    return (g_sum ** 2) / (h_sum + reg_lambda)


def optimal_weight(g_sum: float, h_sum: float, reg_lambda: float) -> float:
    """
    Optimal leaf prediction weight w*.

    This is the closed-form solution obtained by setting
    dObj/dwⱼ = 0.

    Formula:  w* = −Σg / (Σh + λ)

    Parameters
    ----------
    g_sum      : sum of first-order gradients for samples in this leaf
    h_sum      : sum of second-order gradients (hessians) for this leaf
    reg_lambda : L2 regularisation coefficient λ

    Returns
    -------
    float — the optimal prediction value for this leaf
    """
    return -g_sum / (h_sum + reg_lambda)


def split_gain(
    g_left:  float,
    h_left:  float,
    g_right: float,
    h_right: float,
    g_parent: float,
    h_parent: float,
    reg_lambda: float,
    reg_gamma:  float,
) -> float:
    """
    Information gain from splitting a node into two children.

    Formula:
        Gain = ½ [ Score(L) + Score(R) − Score(parent) ] − γ

    A positive gain means the split improves the objective.
    A negative gain (or gain < reg_gamma) means we should not split.

    Parameters
    ----------
    g_left, h_left   : gradient sums for the left child
    g_right, h_right : gradient sums for the right child
    g_parent, h_parent : gradient sums for the parent node
    reg_lambda : L2 regularisation coefficient λ
    reg_gamma  : minimum gain threshold γ

    Returns
    -------
    float — the split gain (use > 0 as the acceptance criterion)
    """
    score_left   = leaf_score(g_left,   h_left,   reg_lambda)
    score_right  = leaf_score(g_right,  h_right,  reg_lambda)
    score_parent = leaf_score(g_parent, h_parent, reg_lambda)
    return 0.5 * (score_left + score_right - score_parent) - reg_gamma
