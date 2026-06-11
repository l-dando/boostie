"""
xgboost_scratch
===============
A from-scratch implementation of the core XGBoost algorithm
using only NumPy and Pandas.

Package layout
--------------
  losses.py      — loss functions and their (g, h) gradients
  math_utils.py  — leaf scoring and optimal weight formulas
  tree.py        — TreeNode and XGBoostTree (single base learner)
  model.py       — XGBoostModel (the full boosting loop)
  data.py        — dataset generation and train/test splitting
  metrics.py     — evaluation metrics (RMSE, log-loss, accuracy…)
  main.py        — end-to-end worked examples
"""

from .model import XGBoostModel
from .tree import XGBoostTree, TreeNode
from .losses import OBJECTIVES, get_objective
from .metrics import rmse, log_loss, accuracy

__all__ = [
    "XGBoostModel",
    "XGBoostTree",
    "TreeNode",
    "OBJECTIVES",
    "get_objective",
    "rmse",
    "log_loss",
    "accuracy",
]
