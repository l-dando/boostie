"""
main.py
=======
End-to-end worked examples for all three supported objectives:

  Example 1 — Regression      (squared-error loss)
  Example 2 — Binary classification  (log-loss)
  Example 3 — Poisson count regression

Run with:
    python main.py

Optionally compare against sklearn (install with `pip install scikit-learn`).
"""

import numpy as np
import pandas as pd

from boostie.model   import XGBoostModel
from boostie.data    import (
    make_regression_data,
    make_classification_data,
    make_count_data,
    train_test_split,
)
from boostie.metrics import (
    rmse, mae, r_squared,
    log_loss, accuracy,
    confusion_matrix, precision_recall,
)


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def subsection(title: str) -> None:
    print(f"\n  --- {title} ---")


# -------------------------------------------------------
# Example 1 — Regression
# -------------------------------------------------------

def example_regression() -> None:
    section("EXAMPLE 1 — Regression (squared-error)")

    # 1. Data
    X, y = make_regression_data(n_samples=600, n_features=6, noise=0.5, seed=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"\n  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # 2. Train
    subsection("Training")
    model = XGBoostModel(
        n_estimators  = 100,
        max_depth     = 4,
        learning_rate = 0.1,
        reg_lambda    = 1.0,
        reg_gamma     = 0.0,
        objective     = "regression",
    )
    model.fit(X_train, y_train, verbose=True, log_every=20)
    print(f"\n  {model}")

    # 3. Evaluate
    subsection("Evaluation")
    preds = model.predict(X_test)
    print(f"  RMSE : {rmse(y_test, preds):.4f}")
    print(f"  MAE  : {mae(y_test, preds):.4f}")
    print(f"  R²   : {r_squared(y_test, preds):.4f}")

    # 4. Feature importances
    subsection("Feature importances")
    importances = model.feature_importances(n_features=X.shape[1])
    for i, imp in enumerate(importances):
        bar = "█" * int(imp * 40)
        print(f"  feature {i}  {imp:.3f}  {bar}")

    # 5. Preview predictions
    subsection("First 8 test predictions")
    df = pd.DataFrame({
        "y_true": y_test[:8].round(3),
        "y_pred": preds[:8].round(3),
        "error":  (preds[:8] - y_test[:8]).round(3),
    })
    print(df.to_string(index=False))

    # 6. Optional sklearn comparison
    _compare_sklearn_regression(model, X_train, y_train, X_test, y_test)


def _compare_sklearn_regression(scratch_model, X_train, y_train, X_test, y_test) -> None:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        subsection("Comparison vs sklearn HistGradientBoostingRegressor")
        ref = HistGradientBoostingRegressor(
            max_iter=100, max_depth=4, learning_rate=0.1
        ).fit(X_train, y_train)
        sk_preds  = ref.predict(X_test)
        our_preds = scratch_model.predict(X_test)
        print(f"  sklearn RMSE : {rmse(y_test, sk_preds):.4f}")
        print(f"  scratch RMSE : {rmse(y_test, our_preds):.4f}")
    except ImportError:
        print("\n  (sklearn not installed — skipping comparison)")


# -------------------------------------------------------
# Example 2 — Binary classification
# -------------------------------------------------------

def example_binary_classification() -> None:
    section("EXAMPLE 2 — Binary Classification (log-loss)")

    # 1. Data
    X, y = make_classification_data(n_samples=600, n_features=6, noise=0.1, seed=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"\n  Train: {X_train.shape}  "
          f"(positive rate: {y_train.mean():.2%})")
    print(f"  Test:  {X_test.shape}  "
          f"(positive rate: {y_test.mean():.2%})")

    # 2. Train
    subsection("Training")
    model = XGBoostModel(
        n_estimators  = 100,
        max_depth     = 3,
        learning_rate = 0.1,
        reg_lambda    = 1.0,
        reg_gamma     = 0.0,
        objective     = "binary",
    )
    model.fit(X_train, y_train, verbose=True, log_every=20)

    # 3. Evaluate
    subsection("Evaluation")
    probs = model.predict(X_test)        # probabilities
    print(f"  Log-loss  : {log_loss(y_test, probs):.4f}")
    print(f"  Accuracy  : {accuracy(y_test, probs):.4f}")

    pr = precision_recall(y_test, probs)
    print(f"  Precision : {pr['precision']:.4f}")
    print(f"  Recall    : {pr['recall']:.4f}")
    print(f"  F1        : {pr['f1']:.4f}")

    # 4. Confusion matrix
    subsection("Confusion matrix (threshold = 0.5)")
    cm = confusion_matrix(y_test, probs)
    print(f"           pred 0   pred 1")
    print(f"  actual 0  {cm[0,0]:>5}    {cm[0,1]:>5}")
    print(f"  actual 1  {cm[1,0]:>5}    {cm[1,1]:>5}")

    # 5. Probability calibration preview
    subsection("Predicted probability distribution (10 bins)")
    bins = np.linspace(0, 1, 11)
    counts, _ = np.histogram(probs, bins=bins)
    for i, c in enumerate(counts):
        lo, hi = bins[i], bins[i+1]
        bar = "▪" * (c // 2)
        print(f"  [{lo:.1f}–{hi:.1f}]  {c:>4}  {bar}")

    _compare_sklearn_classification(X_train, y_train, X_test, y_test)


def _compare_sklearn_classification(X_train, y_train, X_test, y_test) -> None:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        subsection("Comparison vs sklearn HistGradientBoostingClassifier")
        ref = HistGradientBoostingClassifier(
            max_iter=100, max_depth=3, learning_rate=0.1
        ).fit(X_train, y_train)
        sk_acc = np.mean(ref.predict(X_test) == y_test)
        print(f"  sklearn accuracy : {sk_acc:.4f}")
    except ImportError:
        print("\n  (sklearn not installed — skipping comparison)")


# -------------------------------------------------------
# Example 3 — Poisson count regression
# -------------------------------------------------------

def example_poisson() -> None:
    section("EXAMPLE 3 — Poisson Count Regression")

    # 1. Data
    X, y = make_count_data(n_samples=600, n_features=5, seed=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"\n  Train: {X_train.shape}  "
          f"(mean count: {y_train.mean():.2f})")
    print(f"  Test:  {X_test.shape}   "
          f"(mean count: {y_test.mean():.2f})")

    # 2. Train
    subsection("Training")
    model = XGBoostModel(
        n_estimators  = 100,
        max_depth     = 3,
        learning_rate = 0.05,
        reg_lambda    = 1.0,
        objective     = "poisson",
        base_score    = float(np.log(y_train.mean() + 1e-6)),
    )
    model.fit(X_train, y_train, verbose=True, log_every=20)

    # 3. Evaluate  (predict returns expected counts via exp link)
    subsection("Evaluation")
    preds = model.predict(X_test)
    print(f"  RMSE : {rmse(y_test, preds):.4f}")
    print(f"  MAE  : {mae(y_test, preds):.4f}")

    subsection("First 8 test predictions")
    df = pd.DataFrame({
        "y_true":      y_test[:8].astype(int),
        "y_pred (μ)":  preds[:8].round(2),
    })
    print(df.to_string(index=False))


# -------------------------------------------------------
# Entry point
# -------------------------------------------------------

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║        XGBoost From Scratch — Worked Examples            ║
║  objectives: regression | binary | poisson               ║
╚══════════════════════════════════════════════════════════╝
""")
    example_regression()
    example_binary_classification()
    example_poisson()

    print("""
╔══════════════════════════════════════════════════════════╗
║  HYPERPARAMETER GUIDE                                    ║
╠══════════════════════════════════════════════════════════╣
║  n_estimators     More trees = more capacity.            ║
║                   Risk: overfitting + slow training.     ║
║                   Typical: 100–1000.                     ║
║                                                          ║
║  max_depth        Shallower = less overfit.              ║
║                   XGBoost default: 6. Start with 3–5.    ║
║                                                          ║
║  learning_rate    Smaller → more robust, needs more      ║
║  (eta)            trees. Classic: 0.01–0.1.              ║
║                                                          ║
║  reg_lambda (λ)   L2 on leaf weights. Smooths model.     ║
║                   Default: 1.0. Try 0.1–10.              ║
║                                                          ║
║  reg_gamma  (γ)   Minimum gain to accept a split.        ║
║                   Default: 0. Increase to prune more.    ║
║                                                          ║
║  min_child_weight Minimum Σh in a child. Guards against  ║
║                   splits on tiny / noisy samples.        ║
╠══════════════════════════════════════════════════════════╣
║  WHAT THE REAL XGBOOST ADDS                              ║
╠══════════════════════════════════════════════════════════╣
║  • Column (feature) subsampling per tree / per level     ║
║  • Row subsampling (subsample parameter)                 ║
║  • Sparsity-aware split finding (handles NaN natively)   ║
║  • Approximate algorithm (quantile sketching) for        ║
║    very large datasets                                   ║
║  • GPU acceleration & distributed computing              ║
║  • Built-in early stopping & cross-validation            ║
╚══════════════════════════════════════════════════════════╝
""")
