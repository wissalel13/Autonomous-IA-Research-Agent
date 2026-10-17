import tempfile
import time
from pathlib import Path

import joblib
import mlflow
import optuna
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from config import setup_mlflow

from schemas.plan import DataAnalysisPlan
from schemas.results import ModelResult
from tools.modeling import build_pipeline, get_scoring

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest(trial: optuna.Trial, name: str, task: str = "classification") -> dict:
    if name == "logistic_regression":
        return {"model__C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if name == "ridge":
        return {"model__alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True)}
    if name == "random_forest":
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "model__max_depth": trial.suggest_int("max_depth", 3, 12),
            "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    if name == "extra_trees":
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "model__max_depth": trial.suggest_int("max_depth", 3, 15),
            "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    if name == "xgboost":
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "model__max_depth": trial.suggest_int("max_depth", 2, 6),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "model__subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
    if name == "lightgbm":
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "model__num_leaves": trial.suggest_int("num_leaves", 4, 31),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "model__min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
        }
    if name == "hist_gradient_boosting":
        return {
            "model__max_iter": trial.suggest_int("max_iter", 100, 300),
            "model__max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 63),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 40),
        }
    if name == "svm":
        c_low, c_high = (1.0, 1e4) if task == "regression" else (1e-2, 1e2)
        return {
            "model__C": trial.suggest_float("C", c_low, c_high, log=True),
            "model__gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        }
    if name == "knn":
        return {
            "model__n_neighbors": trial.suggest_int("n_neighbors", 3, 40),
            "model__weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
    raise ValueError(f"Pas d'espace de recherche pour {name}")


def _split(df: pd.DataFrame, target: str, task: str, plan: DataAnalysisPlan, seed: int):
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
    return X_train, X_test, y_train, y_test, cv


def tune_model(
    df: pd.DataFrame,
    target: str,
    task: str,
    plan: DataAnalysisPlan,
    model_name: str,
    n_trials: int = 15,
    seed: int = 42,
    experiment_name: str = "ai-research-agent",
) -> ModelResult:
    X_train, X_test, y_train, y_test, cv = _split(df, target, task, plan, seed)
    multiclass = task == "classification" and y_train.nunique() > 2
    scoring = get_scoring(plan.primary_metric, multiclass)
    sign = -1 if scoring.startswith("neg_") else 1

    def make_pipe():
        return build_pipeline(X_train, model_name, task, plan, y_train.to_numpy())

    def objective(trial: optuna.Trial) -> float:
        pipe = make_pipe()
        pipe.set_params(**_suggest(trial, model_name, task))
        # score brut du scorer sklearn : toujours « plus grand = meilleur »
        return float(cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring).mean())

    setup_mlflow(experiment_name)

    with mlflow.start_run(run_name=f"{model_name}_tuned") as run:
        t = time.time()
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=n_trials)

        best_params = {f"model__{k}": v for k, v in study.best_params.items()}
        pipe = make_pipe()
        pipe.set_params(**best_params)
        cv_scores = sign * cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
        pipe.fit(X_train, y_train)
        test_score = float(sign * get_scorer(scoring)(pipe, X_test, y_test))
        fit_seconds = time.time() - t

        mlflow.log_params(
            {"model": model_name, "tuned": True, "n_trials": n_trials, **study.best_params}
        )
        mlflow.log_metrics(
            {
                f"cv_{plan.primary_metric}_mean": float(cv_scores.mean()),
                f"cv_{plan.primary_metric}_std": float(cv_scores.std()),
                f"test_{plan.primary_metric}": test_score,
                "fit_seconds": fit_seconds,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{model_name}_tuned.joblib"
            joblib.dump(pipe, path)
            mlflow.log_artifact(str(path), artifact_path="model")

        return ModelResult(
            model_name=f"{model_name}_tuned",
            cv_mean=round(float(cv_scores.mean()), 4),
            cv_std=round(float(cv_scores.std()), 4),
            test_score=round(test_score, 4),
            fit_seconds=round(fit_seconds, 1),
            run_id=run.info.run_id,
            params=study.best_params,
        )