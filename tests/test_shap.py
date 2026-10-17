import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from schemas.plan import DataAnalysisPlan
from tools.explain import explain_model, load_pipeline
from tools.modeling import run_experiments
from tools.tuning import _split

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

    res = run_experiments(
        df, "target", "classification", plan, ["logistic_regression", "random_forest"]
    )
    X_train, X_test, _, _, _ = _split(df, "target", "classification", plan, 42)

    for r in res.results:
        pipe = load_pipeline(r.run_id, r.model_name)
        e = explain_model(pipe, X_train, X_test, r.model_name, run_id=r.run_id)
        print(f"\n=== {r.model_name} ===")
        for f in e.top_features:
            print(f"  {f.feature:<8} {f.mean_abs_shap:.4f}  {f.direction}")
        print(f"  graphique : {e.plot_path}")