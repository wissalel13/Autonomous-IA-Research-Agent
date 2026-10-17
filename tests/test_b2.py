import pandas as pd
from sklearn.datasets import make_classification, make_regression

from agents.ml_agent import eligible_models
from schemas.plan import DataAnalysisPlan
from tools.explain import explain_model, load_pipeline
from tools.modeling import run_experiments
from tools.profiling import profile_dataset
from tools.tuning import _split


def make_df(n, n_classes=2, weights=None):
    X, y = make_classification(
        n_samples=n, n_features=8, n_informative=5,
        n_classes=n_classes, weights=weights, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["target"] = y
    return df


def make_plan(metric, strategy):
    return DataAnalysisPlan(
        columns_to_drop=[], numeric_imputation="median",
        imbalance_strategy=strategy, primary_metric=metric, reasoning="test",
    )


def shap_check(df, task, plan, models):
    res = run_experiments(df, "target", task, plan, models)
    X_train, X_test, _, _, _ = _split(df, "target", task, plan, 42)
    for r in res.results:
        pipe = load_pipeline(r.run_id, r.model_name)
        e = explain_model(pipe, X_train, X_test, r.model_name, run_id=r.run_id, task=task)
        top = [(f.feature, f.mean_abs_shap, f.direction) for f in e.top_features[:2]]
        print(f"  {task:<14} {r.model_name:<24} {top}")


if __name__ == "__main__":
    print("=== Éligibilité ===")
    cases = [
        ("petit binaire déséquilibré + class_weight", make_df(2000, weights=[0.95, 0.05]), make_plan("average_precision", "class_weight")),
        ("grand binaire (6000 lignes)", make_df(6000), make_plan("f1", "none")),
        ("multiclasse + roc_auc", make_df(2000, 3), make_plan("roc_auc", "none")),
    ]
    for title, df, pl in cases:
        pool, excl = eligible_models(profile_dataset(df, "target"), pl)
        print(f"{title}\n  pool   : {pool}\n  exclus : {excl}")

    print("\n=== SHAP sur les modèles non natifs (peut prendre 1 à 2 minutes) ===")
    shap_check(make_df(2000, weights=[0.9, 0.1]), "classification",
               make_plan("average_precision", "none"), ["svm", "knn", "hist_gradient_boosting"])
    shap_check(make_df(2000, 3), "classification",
               make_plan("balanced_accuracy", "none"), ["svm"])

    Xr, yr = make_regression(n_samples=1500, n_features=6, noise=10, random_state=0)
    dfr = pd.DataFrame(Xr, columns=[f"f{i}" for i in range(6)])
    dfr["target"] = yr
    shap_check(dfr, "regression", make_plan("rmse", "none"), ["svm"])