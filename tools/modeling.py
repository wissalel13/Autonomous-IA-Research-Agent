import tempfile
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.utils.class_weight import compute_sample_weight

from schemas.plan import DataAnalysisPlan
from schemas.results import ExperimentResults, ModelResult
from config import setup_mlflow

SCORERS = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}
MULTICLASS_SCORERS = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1_macro",
    "roc_auc": "roc_auc_ovr",
}

LOWER_IS_BETTER = {"rmse", "mae"}

CLASSIFICATION_MODELS = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
REGRESSION_MODELS = ["ridge", "random_forest", "xgboost", "lightgbm"]
EXTRA_MODELS = ["extra_trees", "hist_gradient_boosting", "svm", "knn"]


def get_scoring(metric: str, multiclass: bool = False) -> str:
    if multiclass:
        if metric not in MULTICLASS_SCORERS:
            raise ValueError(f"Métrique '{metric}' indisponible en multiclasse")
        return MULTICLASS_SCORERS[metric]
    return SCORERS[metric]


class BalancedXGBClassifier(XGBClassifier):
    """XGBoost multiclasse avec poids équilibrés, recalculés sur chaque fold d'entraînement."""

    def fit(self, X, y, **kwargs):
        kwargs["sample_weight"] = compute_sample_weight("balanced", y)
        return super().fit(X, y, **kwargs)


def make_model(name: str, task: str, imbalance: str, y_train: np.ndarray):
    balanced = imbalance == "class_weight"
    cw = "balanced" if balanced else None

    if task == "classification":
        if name == "logistic_regression":
            return LogisticRegression(max_iter=1000, class_weight=cw)
        if name == "random_forest":
            return RandomForestClassifier(
                n_estimators=200, class_weight=cw, random_state=42, n_jobs=-1
            )
        if name == "xgboost":
            counts = np.bincount(y_train)
            if len(counts) > 2:  # multiclasse
                cls = BalancedXGBClassifier if balanced else XGBClassifier
                return cls(n_estimators=200, random_state=42, n_jobs=-1)
            kw = {}
            if balanced:
                kw["scale_pos_weight"] = counts[0] / counts[1]
            return XGBClassifier(
                n_estimators=200, eval_metric="logloss", random_state=42, n_jobs=-1, **kw
            )
        if name == "lightgbm":
            return LGBMClassifier(
                n_estimators=200, class_weight=cw, random_state=42, verbose=-1
            )
        if name == "extra_trees":
            return ExtraTreesClassifier(
                n_estimators=200, class_weight=cw, random_state=42, n_jobs=-1
            )
        if name == "hist_gradient_boosting":
            return HistGradientBoostingClassifier(class_weight=cw, random_state=42)
        if name == "svm":
            return SVC(class_weight=cw, random_state=42)
        if name == "knn":
            return KNeighborsClassifier(n_neighbors=15)  # pas de class_weight
    else:
        if name == "ridge":
            return Ridge()
        if name == "random_forest":
            return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        if name == "xgboost":
            return XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        if name == "lightgbm":
            return LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)
        if name == "extra_trees":
            return ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        if name == "hist_gradient_boosting":
            return HistGradientBoostingRegressor(random_state=42)
        if name == "svm":
            mu, sigma = float(np.mean(y_train)), float(np.std(y_train))
            return SVR(
                C=max(abs(mu + 3 * sigma), abs(mu - 3 * sigma), 1.0),
                epsilon=0.1 * sigma if sigma > 0 else 0.1,
            )
        if name == "knn":
            return KNeighborsRegressor(n_neighbors=15)
    raise ValueError(f"Modèle inconnu : {name} ({task})")


def build_pipeline(X: pd.DataFrame, name: str, task: str, plan: DataAnalysisPlan, y_train: np.ndarray):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    # "none" est traité comme "median" : la plupart des modèles sklearn n'acceptent pas les NaN
    imputation = "mean" if plan.numeric_imputation == "mean" else "median"

    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy=imputation)), ("sc", StandardScaler())]),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    steps = [("prep", prep)]
    if task == "classification":
        if plan.imbalance_strategy == "smote":
            steps.append(("sampler", SMOTE(random_state=42)))
        elif plan.imbalance_strategy == "undersampling":
            steps.append(("sampler", RandomUnderSampler(random_state=42)))
    steps.append(("model", make_model(name, task, plan.imbalance_strategy, y_train)))
    return Pipeline(steps)


def run_experiments(
    df: pd.DataFrame,
    target: str,
    task: str,
    plan: DataAnalysisPlan,
    model_names: list[str],
    experiment_name: str = "ai-research-agent",
    seed: int = 42,
) -> ExperimentResults:
    df = df.drop(columns=plan.columns_to_drop)
    X, y = df.drop(columns=[target]), df[target]

    stratify = None
    if task == "classification":
        y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=stratify, random_state=seed
    )
    cv = (
        StratifiedKFold(5, shuffle=True, random_state=seed)
        if task == "classification"
        else KFold(5, shuffle=True, random_state=seed)
    )

    multiclass = task == "classification" and y_train.nunique() > 2
    scoring = get_scoring(plan.primary_metric, multiclass)
    sign = -1 if scoring.startswith("neg_") else 1

    setup_mlflow(experiment_name)

    results: list[ModelResult] = []
    for name in model_names:
        pipe = build_pipeline(X_train, name, task, plan, y_train.to_numpy())

        with mlflow.start_run(run_name=name) as run:
            t = time.time()
            cv_scores = sign * cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
            pipe.fit(X_train, y_train)
            test_score = float(sign * get_scorer(scoring)(pipe, X_test, y_test))
            fit_seconds = time.time() - t

            mlflow.log_params(
                {
                    "model": name,
                    "task": task,
                    "primary_metric": plan.primary_metric,
                    "imbalance_strategy": plan.imbalance_strategy,
                    "numeric_imputation": plan.numeric_imputation,
                    "dropped_columns": ",".join(plan.columns_to_drop) or "none",
                    "n_features": X_train.shape[1],
                }
            )
            mlflow.log_metrics(
                {
                    f"cv_{plan.primary_metric}_mean": float(cv_scores.mean()),
                    f"cv_{plan.primary_metric}_std": float(cv_scores.std()),
                    f"test_{plan.primary_metric}": test_score,
                    "fit_seconds": fit_seconds,
                }
            )
            mlflow.log_text(plan.model_dump_json(indent=2), "plan.json")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"{name}.joblib"
                joblib.dump(pipe, path)
                mlflow.log_artifact(str(path), artifact_path="model")

            results.append(
                ModelResult(
                    model_name=name,
                    cv_mean=round(float(cv_scores.mean()), 4),
                    cv_std=round(float(cv_scores.std()), 4),
                    test_score=round(test_score, 4),
                    fit_seconds=round(fit_seconds, 1),
                    run_id=run.info.run_id,
                )
            )

    pick = min if plan.primary_metric in LOWER_IS_BETTER else max
    best = pick(results, key=lambda r: r.cv_mean).model_name
    return ExperimentResults(primary_metric=plan.primary_metric, results=results, best_model=best)