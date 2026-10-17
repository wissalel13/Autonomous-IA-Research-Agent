import numpy as np
import pandas as pd

from schemas.data import ColumnProfile, DataProfile, TargetProfile


def _f(x) -> float | None:
    """float arrondi, ou None si NaN."""
    return None if x is None or pd.isna(x) else round(float(x), 4)


def _infer_task(y: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    values = y.dropna()
    integer_like = (values % 1 == 0).all()
    return "classification" if integer_like and values.nunique() <= 10 else "regression"


def _profile_target(y: pd.Series, name: str) -> TargetProfile:
    task = _infer_task(y)
    if task == "regression":
        return TargetProfile(name=name, task=task)
    counts = y.value_counts()
    return TargetProfile(
        name=name,
        task=task,
        n_classes=int(len(counts)),
        class_distribution={str(k): round(v / len(y), 4) for k, v in counts.items()},
        imbalance_ratio=round(float(counts.max() / counts.min()), 2),
    )


def profile_dataset(
    df: pd.DataFrame,
    target: str,
    corr_threshold: float = 0.95,
    max_cardinality: int = 50,
) -> DataProfile:
    X = df.drop(columns=[target])

    columns = []
    for col in X.columns:
        s = X[col]
        p = ColumnProfile(
            name=col,
            dtype=str(s.dtype),
            missing_pct=round(float(s.isna().mean() * 100), 2),
            n_unique=int(s.nunique()),
        )
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            p.mean, p.std = _f(s.mean()), _f(s.std())
            p.min, p.max, p.skew = _f(s.min()), _f(s.max()), _f(s.skew())
        columns.append(p)

    constant = [c.name for c in columns if c.n_unique <= 1]
    high_card = [
        c.name
        for c in columns
        if c.dtype in ("object", "category") and c.n_unique > max_cardinality
    ]

    pairs = []
    num = X.select_dtypes(include="number")
    if num.shape[1] > 1:
        corr = num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        for a in upper.columns:
            for b in upper.index:
                v = upper.loc[b, a]
                if pd.notna(v) and v >= corr_threshold:
                    pairs.append((b, a, round(float(v), 3)))

    return DataProfile(
        n_rows=len(df),
        n_cols=X.shape[1],
        n_duplicates=int(df.duplicated().sum()),
        columns=columns,
        target=_profile_target(df[target], target),
        constant_columns=constant,
        high_cardinality_columns=high_card,
        highly_correlated_pairs=pairs,
    )