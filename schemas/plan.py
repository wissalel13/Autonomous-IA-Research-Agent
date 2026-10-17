from typing import Literal

from pydantic import BaseModel, Field, field_validator

CLASSIFICATION_METRICS = {"accuracy", "f1", "roc_auc", "average_precision", "balanced_accuracy"}
MULTICLASS_METRICS = {"accuracy", "f1", "roc_auc", "balanced_accuracy"}
REGRESSION_METRICS = {"rmse", "mae", "r2"}


class DataAnalysisPlan(BaseModel):
    columns_to_drop: list[str] = Field(default_factory=list)
    numeric_imputation: Literal["mean", "median", "none"]
    imbalance_strategy: Literal["none", "class_weight", "smote", "undersampling"]
    primary_metric: Literal[
        "accuracy", "f1", "roc_auc", "average_precision", "balanced_accuracy",
        "rmse", "mae", "r2",
    ]
    reasoning: str
    corrections: list[str] = Field(default_factory=list)

    @field_validator("reasoning")
    @classmethod
    def _truncate(cls, v: str) -> str:
        v = v[:400]
        return v[: v.rfind(".") + 1] if "." in v else v

class MLPlan(BaseModel):
    models: list[
        Literal[
            "logistic_regression", "ridge", "random_forest", "extra_trees",
            "xgboost", "lightgbm", "hist_gradient_boosting", "svm", "knn",
        ]
    ]
    reasoning: str
    corrections: list[str] = Field(default_factory=list)

    @field_validator("reasoning")
    @classmethod
    def _truncate(cls, v: str) -> str:
        v = v[:400]
        return v[: v.rfind(".") + 1] if "." in v else v

class NextAction(BaseModel):
    action: Literal["optimize", "add_models", "stop"]
    target_model: str | None = None
    reasoning: str
    corrections: list[str] = Field(default_factory=list)

    @field_validator("reasoning")
    @classmethod
    def _truncate(cls, v: str) -> str:
        v = v[:400]
        return v[: v.rfind(".") + 1] if "." in v else v