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

from boostie.model import boostieModel
from boostie.data import (
    make_regression_data,
    make_classification_data,
    make_count_data,
    train_test_split,
)
from boostie.metrics import (
    rmse,
    mae,
    r_squared,
    log_loss,
    accuracy,
    confusion_matrix,
    precision_recall,
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
