"""
========================================================
 XGBoost From Scratch — Step-by-Step Implementation
 Using only NumPy and Pandas
========================================================

OVERVIEW OF STEPS
-----------------
 1. Understand what XGBoost is optimising
 2. Define the loss function and its gradients
 3. Build a single regression tree (the base learner)
 4. Implement the XGBoost tree-growing algorithm
 5. Build the boosting loop
 6. Add regularisation (lambda, gamma)
 7. Add learning rate (eta / shrinkage)
 8. Run a worked example on a toy dataset
 9. Evaluate and compare to sklearn's XGBoost

========================================================
"""

import numpy as np
import pandas as pd
from typing import Optional


# -------------------------------------------------------
# STEP 1 — WHAT IS XGBOOST OPTIMISING?
# -------------------------------------------------------
# XGBoost minimises a regularised objective:
#
#   Obj = Σ L(yᵢ, ŷᵢ)  +  Σ Ω(fₖ)
#         loss              regularisation over each tree
#
# For regression we use the squared-error loss:
#   L(y, ŷ) = ½ (y - ŷ)²
#
# The key insight: rather than optimising the full
# objective directly, XGBoost uses a **second-order
# Taylor expansion** around the current predictions.
# This lets us find the optimal leaf weights and best
# splits analytically — no gradient descent needed.
#
# At each boosting round t, we need:
#   gᵢ = ∂L/∂ŷᵢ   (first-order gradient, called "g")
#   hᵢ = ∂²L/∂ŷᵢ² (second-order gradient, called "h")
#
# For squared-error:
#   gᵢ = ŷᵢ - yᵢ   (the residual)
#   hᵢ = 1          (constant)


# -------------------------------------------------------
# STEP 2 — LOSS FUNCTION AND GRADIENTS
# -------------------------------------------------------

def squared_error_gradients(y: np.ndarray, y_pred: np.ndarray):
    """
    Returns first (g) and second (h) order gradients
    for the mean squared error loss: L = ½(y - ŷ)²
    """
    g = y_pred - y          # dL/dŷ
    h = np.ones_like(y)     # d²L/dŷ² = 1 always
    return g, h


def log_loss_gradients(y: np.ndarray, y_pred: np.ndarray):
    """
    Gradients for binary cross-entropy (logistic) loss.
    y_pred here is the raw margin (log-odds), not a probability.
    """
    prob = 1.0 / (1.0 + np.exp(-y_pred))   # sigmoid
    g = prob - y
    h = prob * (1.0 - prob)
    return g, h


# -------------------------------------------------------
# STEP 3 — THE CORE MATHS: OPTIMAL LEAF WEIGHT & SCORE
# -------------------------------------------------------
# After the Taylor expansion, the gain from putting a
# set of samples S into one leaf is:
#
#   Score(S) = (Σgᵢ)² / (Σhᵢ + λ)
#
# where λ (reg_lambda) is L2 regularisation on leaf weights.
#
# The optimal weight for leaf S is:
#   w* = -Σgᵢ / (Σhᵢ + λ)
#
# The gain from splitting a node into left (L) and right (R):
#   Gain = ½ [Score(L) + Score(R) - Score(parent)] - γ
#
# where γ (reg_gamma) is the minimum gain required to split.


def leaf_score(g_sum: float, h_sum: float, reg_lambda: float) -> float:
    """Structural score of a leaf node."""
    return (g_sum ** 2) / (h_sum + reg_lambda)


def optimal_weight(g_sum: float, h_sum: float, reg_lambda: float) -> float:
    """Optimal prediction value for a leaf."""
    return -g_sum / (h_sum + reg_lambda)


# -------------------------------------------------------
# STEP 4 — BUILD A SINGLE XGBOOST TREE
# -------------------------------------------------------

class TreeNode:
    """A node in our decision tree."""
    def __init__(self):
        self.feature_index: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional['TreeNode'] = None
        self.right: Optional['TreeNode'] = None
        self.leaf_value: Optional[float] = None    # set only for leaf nodes

    def is_leaf(self) -> bool:
        return self.leaf_value is not None


class XGBoostTree:
    """
    A single regression tree trained using XGBoost's
    exact greedy algorithm.

    Parameters
    ----------
    max_depth   : maximum tree depth
    reg_lambda  : L2 regularisation on leaf weights (λ)
    reg_gamma   : minimum gain to make a split (γ)
    min_child_weight : minimum sum of h in a child node
    """

    def __init__(
        self,
        max_depth: int = 3,
        reg_lambda: float = 1.0,
        reg_gamma: float = 0.0,
        min_child_weight: float = 1.0,
    ):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.min_child_weight = min_child_weight
        self.root: Optional[TreeNode] = None

    def fit(self, X: np.ndarray, g: np.ndarray, h: np.ndarray):
        """Grow the tree on gradients g and hessians h."""
        self.root = self._grow(X, g, h, depth=0)

    def _grow(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        depth: int,
    ) -> TreeNode:
        node = TreeNode()

        # --- Stopping conditions ---
        if depth >= self.max_depth or len(g) <= 1:
            node.leaf_value = optimal_weight(g.sum(), h.sum(), self.reg_lambda)
            return node

        # --- Find the best split ---
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        best_left_mask = None

        parent_score = leaf_score(g.sum(), h.sum(), self.reg_lambda)
        n_features = X.shape[1]

        for feat in range(n_features):
            # Sort samples by this feature
            order = np.argsort(X[:, feat])
            X_sorted = X[order, feat]
            g_sorted = g[order]
            h_sorted = h[order]

            # Accumulate left sums as we scan thresholds
            g_left, h_left = 0.0, 0.0
            g_right, h_right = g_sorted.sum(), h_sorted.sum()

            for i in range(len(g_sorted) - 1):
                g_left  += g_sorted[i];  h_left  += h_sorted[i]
                g_right -= g_sorted[i];  h_right -= h_sorted[i]

                # Skip if adjacent values are identical (no valid split point)
                if X_sorted[i] == X_sorted[i + 1]:
                    continue

                # Enforce min_child_weight
                if h_left < self.min_child_weight or h_right < self.min_child_weight:
                    continue

                gain = 0.5 * (
                    leaf_score(g_left, h_left, self.reg_lambda)
                    + leaf_score(g_right, h_right, self.reg_lambda)
                    - parent_score
                ) - self.reg_gamma

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = (X_sorted[i] + X_sorted[i + 1]) / 2.0
                    best_left_mask = X[:, feat] <= best_threshold

        # --- No beneficial split found: make a leaf ---
        if best_feature is None:
            node.leaf_value = optimal_weight(g.sum(), h.sum(), self.reg_lambda)
            return node

        # --- Recurse ---
        node.feature_index = best_feature
        node.threshold = best_threshold

        left_mask  = best_left_mask
        right_mask = ~left_mask

        node.left  = self._grow(X[left_mask],  g[left_mask],  h[left_mask],  depth + 1)
        node.right = self._grow(X[right_mask], g[right_mask], h[right_mask], depth + 1)
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return leaf values for every sample in X."""
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x: np.ndarray, node: TreeNode) -> float:
        if node.is_leaf():
            return node.leaf_value
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


# -------------------------------------------------------
# STEP 5, 6, 7 — THE BOOSTING LOOP WITH REGULARISATION
#                AND LEARNING RATE
# -------------------------------------------------------
# Boosting = iteratively fit trees to the *residual signal*
# encoded by (g, h).
#
# Each round t:
#   1. Compute g, h from current predictions ŷ
#   2. Fit a new tree fₜ to (g, h)
#   3. Update: ŷ ← ŷ + η * fₜ(x)
#
# η (learning rate / eta) shrinks each tree's contribution,
# forcing the model to learn more slowly and reducing
# overfitting — the same idea as shrinkage in gradient boosting.


class XGBoostFromScratch:
    """
    Gradient boosted trees built from scratch using only
    NumPy and Pandas — replicating XGBoost's core algorithm.

    Parameters
    ----------
    n_estimators     : number of boosting rounds
    max_depth        : maximum depth per tree
    learning_rate    : η — shrinkage applied to each tree
    reg_lambda       : λ — L2 regularisation on leaf weights
    reg_gamma        : γ — minimum gain required to split
    min_child_weight : minimum sum of hessians in a child
    objective        : 'regression' or 'binary'
    base_score       : initial prediction for all samples
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        reg_lambda: float = 1.0,
        reg_gamma: float = 0.0,
        min_child_weight: float = 1.0,
        objective: str = 'regression',
        base_score: float = 0.5,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.min_child_weight = min_child_weight
        self.objective = objective
        self.base_score = base_score
        self.trees: list[XGBoostTree] = []

    # --- choose gradient function based on objective ---
    def _gradients(self, y, y_pred):
        if self.objective == 'binary':
            return log_loss_gradients(y, y_pred)
        return squared_error_gradients(y, y_pred)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        """Train the model."""
        self.trees = []

        # Initialise predictions with the base score
        y_pred = np.full_like(y, fill_value=self.base_score, dtype=float)

        for t in range(self.n_estimators):
            # Step 5: compute gradients from current predictions
            g, h = self._gradients(y, y_pred)

            # Step 6 & 7: fit a regularised tree and shrink it
            tree = XGBoostTree(
                max_depth=self.max_depth,
                reg_lambda=self.reg_lambda,
                reg_gamma=self.reg_gamma,
                min_child_weight=self.min_child_weight,
            )
            tree.fit(X, g, h)
            update = self.learning_rate * tree.predict(X)

            # Update predictions
            y_pred += update
            self.trees.append(tree)

            if verbose and (t + 1) % 10 == 0:
                g_now, _ = self._gradients(y, y_pred)
                rmse = np.sqrt(np.mean(g_now ** 2))
                print(f"  Round {t+1:>4d}  |  gradient RMSE: {rmse:.6f}")

        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Return raw margin scores (before sigmoid for binary)."""
        y_pred = np.full(X.shape[0], fill_value=self.base_score)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return final predictions:
          - regression: raw scores
          - binary:     probabilities via sigmoid
        """
        raw = self.predict_raw(X)
        if self.objective == 'binary':
            return 1.0 / (1.0 + np.exp(-raw))
        return raw


# -------------------------------------------------------
# STEP 8 — WORKED EXAMPLE: REGRESSION ON A TOY DATASET
# -------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print(" XGBoost From Scratch — Worked Example")
    print("=" * 60)

    # --- 8a. Create a toy regression dataset ---
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 5))
    # True relationship: y depends non-linearly on features 0 and 1
    y = (
        2 * X[:, 0]
        - 1.5 * X[:, 1] ** 2
        + 0.5 * X[:, 2]
        + rng.standard_normal(n) * 0.5   # noise
    )

    # Train / test split (80/20)
    split = int(0.8 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --- 8b. Train our scratch model ---
    print("\n[1] Training XGBoost from scratch ...")
    model = XGBoostFromScratch(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        reg_lambda=1.0,
        reg_gamma=0.0,
        min_child_weight=1.0,
        objective='regression',
        base_score=float(y_train.mean()),
    )
    model.fit(X_train, y_train, verbose=True)

    preds = model.predict(X_test)
    rmse_scratch = np.sqrt(np.mean((preds - y_test) ** 2))
    print(f"\n  Scratch model test RMSE: {rmse_scratch:.4f}")

    # --- 8c. Compare to sklearn's HistGradientBoosting (XGBoost-like) ---
    print("\n[2] Comparing to sklearn HistGradientBoostingRegressor ...")
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        ref = HistGradientBoostingRegressor(
            max_iter=100, max_depth=4, learning_rate=0.1
        ).fit(X_train, y_train)
        rmse_sklearn = np.sqrt(np.mean((ref.predict(X_test) - y_test) ** 2))
        print(f"  sklearn reference test RMSE: {rmse_sklearn:.4f}")
        print(f"\n  Ratio (scratch / sklearn): {rmse_scratch / rmse_sklearn:.3f}")
        print("  (values close to 1.0 mean our implementation is competitive)\n")
    except ImportError:
        print("  sklearn not installed — skipping comparison.\n")

    # --- 8d. Show a results table ---
    results = pd.DataFrame({
        'y_true': y_test[:10],
        'y_pred_scratch': preds[:10],
    }).round(3)
    print("[3] First 10 test predictions:\n")
    print(results.to_string(index=False))

    # -------------------------------------------------------
    # STEP 9 — KEY HYPERPARAMETERS CHEAT SHEET
    # -------------------------------------------------------
    print("""
====================================================
 HYPERPARAMETER CHEAT SHEET
====================================================

 n_estimators   More trees = more capacity, but slower
                and can overfit. Use early stopping in
                practice. Typical range: 100–1000.

 max_depth       Controls tree complexity. XGBoost
                default is 6. Shallower = less overfit.

 learning_rate   η (eta). Smaller = more robust but
                needs more trees. Classic combo:
                low η (0.01–0.1) + high n_estimators.

 reg_lambda      L2 penalty on leaf weights. Increasing
                smooths the model. Default: 1.0.

 reg_gamma       Minimum gain to create a split (γ).
                Acts as a hard pruning rule.
                Default: 0 (no minimum).

 min_child_weight  Minimum Σh in a leaf. Guards against
                   splits on tiny, noisy samples.

====================================================
 HOW THE REAL XGBOOST GOES FURTHER
====================================================

 - Column (feature) subsampling per tree / per level
 - Row subsampling (subsample parameter)
 - Sparsity-aware split finding (handles missing values)
 - Approximate greedy algorithm (quantile sketching)
   for very large datasets
 - GPU acceleration
 - Distributed computing support
 - Built-in cross-validation and early stopping

====================================================
""")
