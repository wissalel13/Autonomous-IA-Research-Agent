from pydantic import BaseModel


class ModelResult(BaseModel):
    model_name: str
    cv_mean: float
    cv_std: float
    test_score: float
    fit_seconds: float
    run_id: str
    params: dict | None = None


class ExperimentResults(BaseModel):
    primary_metric: str
    results: list[ModelResult]
    best_model: str

class FeatureImportance(BaseModel):
    feature: str
    mean_abs_shap: float
    direction: str


class ExplanationResult(BaseModel):
    model_name: str
    top_features: list[FeatureImportance]
    plot_path: str