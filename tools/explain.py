from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor

from config import MLFLOW_URI
from schemas.results import ExplanationResult, FeatureImportance

NATIVE_MODELS = (
    LogisticRegression, Ridge,
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    XGBClassifier, XGBRegressor, LGBMClassifier, LGBMRegressor,
)


def load_pipeline(run_id: str, model_name: str):
    mlflow.set_tracking_uri(MLFLOW_URI)
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=f"model/{model_name}.joblib"
    )
    return joblib.load(path)


def _direction(x: np.ndarray, s: np.ndarray) -> str:
    if np.std(x) == 0 or np.std(s) == 0:
        return "mixte"
    corr = np.corrcoef(x, s)[0, 1]
    if corr > 0.3:
        return "augmente la prédiction"
    if corr < -0.3:
        return "diminue la prédiction"
    return "mixte"


def explain_model(
    pipe,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
    run_id: str | None = None,
    out_dir: str = "outputs",
    top_k: int = 8,
    max_background: int = 100,
    task: str = "classification",
) -> ExplanationResult:
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]

    Xtr, Xte = prep.transform(X_train), prep.transform(X_test)

    if isinstance(model, NATIVE_MODELS):
        background = shap.sample(Xtr, min(max_background, len(Xtr)), random_state=42)
        sv = shap.Explainer(model, background)(Xte).values
    else:
        # SVM, k-NN, HistGradientBoosting : explication indépendante du modèle,
        # plus lente, donc sur un échantillon réduit
        Xte = Xte[:50]
        background = shap.sample(Xtr, min(25, len(Xtr)), random_state=42)
        if task == "regression":
            fn = model.predict
        elif hasattr(model, "predict_proba"):
            fn = model.predict_proba
        else:
            fn = model.decision_function
        explainer = shap.Explainer(fn, background, algorithm="permutation")
        sv = explainer(Xte, max_evals=max(200, 2 * Xtr.shape[1] + 1)).values

    multiclass = sv.ndim == 3 and sv.shape[2] > 2
    if sv.ndim == 3 and not multiclass:  # binaire : classe positive
        sv = sv[:, :, 1]

    importance = (
        np.abs(sv).mean(axis=(0, 2)) if multiclass else np.abs(sv).mean(axis=0)
    )
    order = np.argsort(importance)[::-1][:top_k]
    top = [
        FeatureImportance(
            feature=names[j],
            mean_abs_shap=round(float(importance[j]), 4),
            direction="variable selon la classe" if multiclass else _direction(Xte[:, j], sv[:, j]),
        )
        for j in order
    ]

    Path(out_dir).mkdir(exist_ok=True)
    plot_path = str(Path(out_dir) / f"shap_{model_name}.png")
    plt.figure(figsize=(7, 4))
    plt.barh([t.feature for t in top][::-1], [t.mean_abs_shap for t in top][::-1])
    plt.xlabel("Importance moyenne |SHAP|")
    plt.title(f"Importance des features : {model_name}")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()

    if run_id:
        mlflow.set_tracking_uri(MLFLOW_URI)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(plot_path, artifact_path="explainability")

    return ExplanationResult(model_name=model_name, top_features=top, plot_path=plot_path)