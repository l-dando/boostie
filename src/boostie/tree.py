"""
tree.py
=======
A single gradient-boosted decision tree — the base learner used
by XGBoostModel.

Classes
-------
  TreeNode     — a single node (either an internal split or a leaf)
  XGBoostTree  — the full tree: builds itself via the exact greedy
                 split-finding algorithm, then predicts leaf values

Algorithm overview (exact greedy)
----------------------------------
For each node at depth d:
  1. If d == max_depth or only one sample remains → make a leaf.
  2. For every feature f:
       Sort samples by f.
       Scan left-to-right, accumulating g_left / h_left and
       decrementing g_right / h_right.
       At each valid threshold, compute split_gain(…).
  3. Accept the (feature, threshold) with the highest positive gain.
  4. Recurse on the left and right subsets.
  5. If no split beats reg_gamma → make a leaf.

Leaf values are the closed-form optimal weights w* from math_utils.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from .math_utils import leaf_score, optimal_weight, split_gain

# -------------------------------------------------------
# TreeNode
# -------------------------------------------------------


class TreeNode:
    """
    One node in the decision tree.

    Internal node : feature_index and threshold are set;
                    left and right children are TreeNodes.
    Leaf node     : leaf_value is set; children are None.
    """

    __slots__ = ("feature_index", "threshold", "left", "right", "leaf_value")

    def __init__(self) -> None:
        self.feature_index: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional[TreeNode] = None
        self.right: Optional[TreeNode] = None
        self.leaf_value: Optional[float] = None

    def is_leaf(self) -> bool:
        """True if this node holds a prediction (no children)."""
        return self.leaf_value is not None

    def __repr__(self) -> str:
        if self.is_leaf():
            return f"TreeNode(leaf={self.leaf_value:.4f})"
        return f"TreeNode(feat={self.feature_index}, " f"thresh={self.threshold:.4f})"


# -------------------------------------------------------
# XGBoostTree
# -------------------------------------------------------


class boosTree:
    """
    A single regression tree trained with XGBoost's exact greedy
    split-finding algorithm.

    The tree is fit to per-sample gradients (g) and hessians (h)
    rather than raw labels, which is what enables the boosting loop
    to work for arbitrary differentiable loss functions.

    Parameters
    ----------
    max_depth        : int, default 3
        Maximum depth of the tree.  Deeper trees can model more
        complex interactions but overfit more easily.
    reg_lambda       : float, default 1.0
        L2 regularisation on leaf weights (λ).  Larger values
        shrink leaf predictions toward zero.
    reg_gamma        : float, default 0.0
        Minimum gain required to accept a split (γ).  Acts as a
        post-pruning threshold.  0 means any positive gain is OK.
    min_child_weight : float, default 1.0
        Minimum sum of hessians required in each child after a
        split.  Prevents splits that affect very few samples.
    """

    def __init__(
        self,
        max_depth: int = 3,
        reg_lambda: float = 1.0,
        reg_gamma: float = 0.0,
        min_child_weight: float = 1.0,
    ) -> None:
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.min_child_weight = min_child_weight
        self.root: Optional[TreeNode] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
    ) -> "XGBoostTree":
        """
        Grow the tree to fit the gradient signal (g, h).

        Parameters
        ----------
        X : feature matrix, shape (n_samples, n_features)
        g : first-order gradients,  shape (n_samples,)
        h : second-order gradients, shape (n_samples,)

        Returns
        -------
        self (for chaining)
        """
        self.root = self._grow(X, g, h, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the leaf value for every row in X.

        Parameters
        ----------
        X : feature matrix, shape (n_samples, n_features)

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
        """
        if self.root is None:
            raise RuntimeError("Tree has not been fitted yet. Call fit() first.")
        return np.array([self._traverse(x, self.root) for x in X])

    # ------------------------------------------------------------------
    # Tree growing (private)
    # ------------------------------------------------------------------

    def _grow(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        depth: int,
    ) -> TreeNode:
        """Recursively grow the tree and return the root node."""
        node = TreeNode()

        # ---- Stopping conditions: make a leaf ----
        if depth >= self.max_depth or len(g) <= 1:
            node.leaf_value = optimal_weight(g.sum(), h.sum(), self.reg_lambda)
            return node

        # ---- Search for the best split ----
        best = self._best_split(X, g, h)

        if best is None:
            # No split improved the objective → leaf
            node.leaf_value = optimal_weight(g.sum(), h.sum(), self.reg_lambda)
            return node

        # ---- Apply the split and recurse ----
        feat_idx, threshold = best
        left_mask = X[:, feat_idx] <= threshold
        right_mask = ~left_mask

        node.feature_index = feat_idx
        node.threshold = threshold
        node.left = self._grow(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        node.right = self._grow(X[right_mask], g[right_mask], h[right_mask], depth + 1)
        return node

    def _best_split(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
    ) -> Optional[tuple[int, float]]:
        """
        Find the (feature_index, threshold) pair that maximises
        split_gain across all features and all candidate thresholds.

        Returns None if no valid split exists.
        """
        best_gain = 0.0  # must beat 0 (already includes -γ via split_gain)
        best_feature = None
        best_threshold = None

        g_parent = g.sum()
        h_parent = h.sum()

        for feat in range(X.shape[1]):
            order = np.argsort(X[:, feat])
            x_sorted = X[order, feat]
            g_sorted = g[order]
            h_sorted = h[order]

            g_left, h_left = 0.0, 0.0
            g_right, h_right = g_parent, h_parent

            for i in range(len(g_sorted) - 1):
                g_left += g_sorted[i]
                h_left += h_sorted[i]
                g_right -= g_sorted[i]
                h_right -= h_sorted[i]

                # No split between identical feature values
                if x_sorted[i] == x_sorted[i + 1]:
                    continue

                # Guard against under-populated children
                if h_left < self.min_child_weight or h_right < self.min_child_weight:
                    continue

                gain = split_gain(
                    g_left,
                    h_left,
                    g_right,
                    h_right,
                    g_parent,
                    h_parent,
                    self.reg_lambda,
                    self.reg_gamma,
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = (x_sorted[i] + x_sorted[i + 1]) / 2.0

        if best_feature is None:
            return None
        return best_feature, best_threshold

    # ------------------------------------------------------------------
    # Prediction (private)
    # ------------------------------------------------------------------

    def _traverse(self, x: np.ndarray, node: TreeNode) -> float:
        """Walk one sample down the tree and return its leaf value."""
        if node.is_leaf():
            return node.leaf_value
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def depth(self) -> int:
        """Return the actual depth of the fitted tree."""

        def _depth(node: Optional[TreeNode]) -> int:
            if node is None or node.is_leaf():
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))

        return _depth(self.root)

    def n_leaves(self) -> int:
        """Return the number of leaf nodes."""

        def _count(node: Optional[TreeNode]) -> int:
            if node is None:
                return 0
            if node.is_leaf():
                return 1
            return _count(node.left) + _count(node.right)

        return _count(self.root)

    def __repr__(self) -> str:
        if self.root is None:
            return "XGBoostTree(unfitted)"
        return (
            f"XGBoostTree(depth={self.depth()}, "
            f"leaves={self.n_leaves()}, "
            f"lambda={self.reg_lambda}, "
            f"gamma={self.reg_gamma})"
        )
