from pydantic import BaseModel


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    missing_pct: float
    n_unique: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    skew: float | None = None


class TargetProfile(BaseModel):
    name: str
    task: str  # "classification" ou "regression"
    n_classes: int | None = None
    class_distribution: dict[str, float] | None = None
    imbalance_ratio: float | None = None


class DataProfile(BaseModel):
    n_rows: int
    n_cols: int
    n_duplicates: int
    columns: list[ColumnProfile]
    target: TargetProfile
    constant_columns: list[str]
    high_cardinality_columns: list[str]
    highly_correlated_pairs: list[tuple[str, str, float]]