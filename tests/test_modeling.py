import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from schemas.plan import DataAnalysisPlan
from tools.modeling import CLASSIFICATION_MODELS, run_experiments

if __name__ == "__main__":
    X, y = make_classification(
        n_samples=2000, n_features=6, weights=[0.95, 0.05], random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    df["const"] = 1
    df["f0_copy"] = df["f0"] * 2 + 0.001
    df.loc[df.sample(frac=0.2, random_state=0).index, "f1"] = np.nan

    plan = DataAnalysisPlan(
        columns_to_drop=["const", "f0_copy"],
        numeric_imputation="median",
        imbalance_strategy="class_weight",
        primary_metric="average_precision",
        reasoning="plan de test",
    )

    res = run_experiments(df, "target", "classification", plan, CLASSIFICATION_MODELS)
    print(f"Métrique : {res.primary_metric}\n")
    for r in res.results:
        print(f"{r.model_name:<20} cv={r.cv_mean:.3f} ± {r.cv_std:.3f}  test={r.test_score:.3f}  ({r.fit_seconds}s)")
    print(f"\nMeilleur modèle : {res.best_model}")