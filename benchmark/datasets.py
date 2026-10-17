import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_wine,
    make_classification,
    make_friedman1,
)


def _frame(X, y, names) -> pd.DataFrame:
    df = pd.DataFrame(X, columns=list(names))
    df["target"] = y
    return df


def cancer() -> pd.DataFrame:
    d = load_breast_cancer()
    return _frame(d.data, d.target, [n.replace(" ", "_") for n in d.feature_names])


def rare_synth() -> pd.DataFrame:
    X, y = make_classification(
        n_samples=4000, n_features=10, n_informative=5, n_redundant=2,
        weights=[0.97, 0.03], flip_y=0.01, random_state=0,
    )
    df = _frame(X, y, [f"f{i}" for i in range(10)])
    rng = np.random.default_rng(0)
    for c in ("f1", "f4", "f7"):  # défauts injectés : valeurs manquantes
        df.loc[rng.random(len(df)) < 0.15, c] = np.nan
    df["const"] = 7  # colonne constante
    df["f2_dup"] = df["f2"] * 3.0 + 0.001  # doublon quasi exact
    return df


def wine() -> pd.DataFrame:
    d = load_wine()
    return _frame(d.data, d.target, d.feature_names)


def digits() -> pd.DataFrame:
    d = load_digits()
    return _frame(d.data, d.target, [f"p{i}" for i in range(64)])


def diabetes() -> pd.DataFrame:
    d = load_diabetes()
    return _frame(d.data, d.target, d.feature_names)


def xor_synth() -> pd.DataFrame:
    """Classification non linéaire : la classe dépend du signe de f0 * f1 (3 % de bruit)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 6))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    flip = rng.random(len(y)) < 0.03
    y = np.where(flip, 1 - y, y)
    return _frame(X, y, [f"f{i}" for i in range(6)])


def friedman() -> pd.DataFrame:
    """Régression non linéaire classique (Friedman n°1), 5 variables informatives sur 10."""
    X, y = make_friedman1(n_samples=1000, n_features=10, noise=1.0, random_state=0)
    return _frame(X, y, [f"f{i}" for i in range(10)])


# metric : métrique d'évaluation, fixée par nous (jamais choisie par l'agent)
DATASETS = {
    "cancer": dict(load=cancer, task="classification", metric="roc_auc"),
    "rare_synth": dict(load=rare_synth, task="classification", metric="average_precision"),
    "wine": dict(load=wine, task="classification", metric="f1"),
    "digits": dict(load=digits, task="classification", metric="f1"),
    "diabetes": dict(load=diabetes, task="regression", metric="rmse"),
    "xor_synth": dict(load=xor_synth, task="classification", metric="roc_auc"),
    "friedman": dict(load=friedman, task="regression", metric="rmse"),
}