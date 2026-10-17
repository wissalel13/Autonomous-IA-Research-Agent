import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from schemas.plan import DataAnalysisPlan
from tools.modeling import EXTRA_MODELS, run_experiments
from tools.tuning import tune_model


def show(title, res):
    print(f"\n=== {title} ({res.primary_metric}) ===")
    for r in res.results:
        print(f"  {r.model_name:<24} cv={r.cv_mean:.3f} ± {r.cv_std:.3f}  test={r.test_score:.3f}  ({r.fit_seconds}s)")


if __name__ == "__main__":
    # 1. binaire déséquilibré (même dataset que d'habitude)
    X, y = make_classification(n_samples=2000, n_features=6, weights=[0.95, 0.05], random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    plan = DataAnalysisPlan(
        columns_to_drop=[], numeric_imputation="median", imbalance_strategy="class_weight",
        primary_metric="average_precision", reasoning="test",
    )
    show("Binaire", run_experiments(df, "target", "classification", plan, EXTRA_MODELS))

    for m in ("svm", "knn"):
        r = tune_model(df, "target", "classification", plan, m, n_trials=5)
        print(f"  {r.model_name:<24} cv={r.cv_mean:.3f} ± {r.cv_std:.3f}  params={r.params}  ({r.fit_seconds}s)")

    # 2. multiclasse
    X, y = make_classification(
        n_samples=2000, n_features=8, n_informative=5, n_classes=3,
        weights=[0.7, 0.2, 0.1], random_state=42,
    )
    df3 = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df3["target"] = y
    plan3 = plan.model_copy(update={"primary_metric": "balanced_accuracy"})
    show("Multiclasse", run_experiments(df3, "target", "classification", plan3, EXTRA_MODELS))

    # 3. régression
    Xr, yr = make_regression(n_samples=1500, n_features=6, noise=10, random_state=0)
    dfr = pd.DataFrame(Xr, columns=[f"f{i}" for i in range(6)])
    dfr["target"] = yr
    planr = plan.model_copy(update={"imbalance_strategy": "none", "primary_metric": "rmse"})
    show("Régression", run_experiments(dfr, "target", "regression", planr, EXTRA_MODELS))