"""
model.py
========
XGBoostModel — the full gradient boosting loop.

How it works
------------
XGBoost builds an ensemble of trees additively.  At each round t:

  1. Compute per-sample gradients g and hessians h from the
     current predictions ŷ and the true labels y, using the
     chosen loss function (objective).

  2. Fit a new tree fₜ to (g, h) — the tree finds the split
     pattern that best reduces the objective.

  3. Update predictions:
         ŷ  ←  ŷ  +  η · fₜ(x)
     where η (learning_rate / eta) is a shrinkage factor that
     prevents any single tree from dominating.

After n_estimators rounds the model has learned to approximate
the target via the sum of many shallow trees, each correcting
the residual errors of those before it.

Link functions (objective → final prediction)
---------------------------------------------
  'regression'  → identity      (raw score = prediction)
  'binary'      → sigmoid       (raw score → probability)
  'poisson'     → exp           (raw score → expected count)

Usage
-----
    from xgboost_scratch.model import XGBoostModel

    model = XGBoostModel(n_estimators=100, max_depth=4,
                         learning_rate=0.1, objective='regression')
    model.fit(X_train, y_train, verbose=True)
    preds = model.predict(X_test)
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from .tree import boosTree
from .losses import get_objective


class boostieModel:
    """
    Gradient boosted trees — from-scratch XGBoost implementation.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of boosting rounds (trees to build).
    max_depth : int, default 3
        Maximum depth of each tree.
    learning_rate : float, default 0.1
        Shrinkage factor η applied to each tree's output.
        Lower values are more conservative and require more trees.
    reg_lambda : float, default 1.0
        L2 regularisation on leaf weights (λ).
    reg_gamma : float, default 0.0
        Minimum gain required to accept a split (γ).
    min_child_weight : float, default 1.0
        Minimum sum of hessians in a child node.
    objective : str, default 'regression'
        Loss function to optimise. One of:
          'regression' — mean squared error
          'binary'     — binary cross-entropy
          'poisson'    — Poisson log-likelihood
    base_score : float or None, default None
        Initial prediction for all samples before any tree is added.
        If None, defaults to mean(y) for regression and 0.0 for others.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        reg_lambda: float = 1.0,
        reg_gamma: float = 0.0,
        min_child_weight: float = 1.0,
        objective: str = "regression",
        base_score: Optional[float] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.min_child_weight = min_child_weight
        self.objective = objective
        self.base_score = base_score

        # Set after fit()
        self._trees: list[boosTree] = []
        self._base_score: float = 0.0
        self._grad_fn = get_objective(objective)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = False,
        log_every: int = 10,
    ) -> "boostieModel":
        """
        Train the model on (X, y).

        Parameters
        ----------
        X         : feature matrix, shape (n_samples, n_features)
        y         : target vector,  shape (n_samples,)
        verbose   : if True, print training loss every `log_every` rounds
        log_every : print interval when verbose=True

        Returns
        -------
        self (for chaining)
        """
        self._trees = []

        # Determine initial prediction
        if self.base_score is not None:
            self._base_score = float(self.base_score)
        elif self.objective == "regression":
            self._base_score = float(np.mean(y))
        else:
            self._base_score = 0.0

        y_pred = np.full(len(y), fill_value=self._base_score, dtype=float)

        for t in range(self.n_estimators):
            g, h = self._grad_fn(y, y_pred)

            tree = boosTree(
                max_depth=self.max_depth,
                reg_lambda=self.reg_lambda,
                reg_gamma=self.reg_gamma,
                min_child_weight=self.min_child_weight,
            )
            tree.fit(X, g, h)

            y_pred += self.learning_rate * tree.predict(X)
            self._trees.append(tree)

            if verbose and (t + 1) % log_every == 0:
                loss = self._training_loss(y, y_pred)
                print(
                    f"  [round {t+1:>4d}/{self.n_estimators}]  "
                    f"train loss: {loss:.6f}"
                )

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw margin scores before the link function.

        Useful for inspecting the model internals or when you want
        to apply your own post-processing.
        """
        self._check_fitted()
        y_pred = np.full(X.shape[0], fill_value=self._base_score, dtype=float)
        for tree in self._trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return final predictions (after the link function).

          objective='regression' → raw scores (identity link)
          objective='binary'     → probabilities in [0, 1]
          objective='poisson'    → expected counts (≥ 0)

        Parameters
        ----------
        X : feature matrix, shape (n_samples, n_features)

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
        """
        raw = self.predict_raw(X)
        return self._apply_link(raw)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        For binary classification: return [P(y=0), P(y=1)] per sample.
        Raises ValueError for non-binary objectives.
        """
        if self.objective != "binary":
            raise ValueError("predict_proba is only available for objective='binary'.")
        prob = self.predict(X)
        return np.column_stack([1 - prob, prob])

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def n_trees(self) -> int:
        """Number of trees actually fitted."""
        return len(self._trees)

    def feature_importances(self, n_features: int) -> np.ndarray:
        """
        Compute feature importances as the total split gain contributed
        by each feature across all trees (a.k.a. 'gain' importance).

        Parameters
        ----------
        n_features : number of features in the training data

        Returns
        -------
        importances : np.ndarray, shape (n_features,), sums to 1
        """
        self._check_fitted()
        counts = np.zeros(n_features, dtype=float)
        for tree in self._trees:
            self._count_splits(tree.root, counts)
        total = counts.sum()
        return counts / total if total > 0 else counts

    def _count_splits(self, node, counts: np.ndarray) -> None:
        """Recursively count how many times each feature is used to split."""
        if node is None or node.is_leaf():
            return
        counts[node.feature_index] += 1
        self._count_splits(node.left, counts)
        self._count_splits(node.right, counts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_link(self, raw: np.ndarray) -> np.ndarray:
        """Apply the inverse link function to raw margin scores."""
        if self.objective == "binary":
            return 1.0 / (1.0 + np.exp(-raw))  # sigmoid
        if self.objective == "poisson":
            return np.exp(raw)  # log link
        return raw  # identity (regression)

    def _training_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """Scalar training loss for logging. Uses MSE for all objectives."""
        g, _ = self._grad_fn(y, y_pred)
        return float(np.mean(g**2))

    def _check_fitted(self) -> None:
        if not self._trees:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

    def __repr__(self) -> str:
        return (
            f"XGBoostModel("
            f"objective={self.objective!r}, "
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"lr={self.learning_rate}, "
            f"lambda={self.reg_lambda}, "
            f"gamma={self.reg_gamma})"
        )
