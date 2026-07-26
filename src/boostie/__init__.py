"""
boostie
=======
A from-scratch implementation of gradient-boosted decision trees inspired by XGBoost,
using only NumPy and Pandas.

Package layout
--------------
  losses.py        — loss functions and their (g, h) gradients
  math_utils.py    — leaf scoring and optimal weight formulas
  preprocessors.py — simple categorical encoders (one-hot / label)
  tree.py          — TreeNode and boosTree (single base learner)
  model.py         — boostieModel (the full boosting loop)
  data.py          — dataset generation and train/test splitting
  metrics.py       — evaluation metrics (RMSE, log-loss, accuracy…)
"""

from .model import boostieModel
from .tree import boosTree, TreeNode
from .losses import OBJECTIVES, get_objective
from .metrics import rmse, log_loss, accuracy

__all__ = [
    "boostieModel",
    "boosTree",
    "TreeNode",
    "OBJECTIVES",
    "get_objective",
    "rmse",
    "log_loss",
    "accuracy",
]


def main() -> None:
    """Console entry point for the `boostie` CLI."""
    print("boostie is a Python package. Import `boostieModel` from `boostie` to use it.")
