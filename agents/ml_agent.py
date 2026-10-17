import pandas as pd

from llm.client import LLMClient
from schemas.data import DataProfile
from schemas.plan import DataAnalysisPlan, MLPlan
from schemas.results import ExperimentResults
from tools.modeling import (
    CLASSIFICATION_MODELS,
    EXTRA_MODELS,
    REGRESSION_MODELS,
    run_experiments,
)

SVM_MAX_ROWS = 5000

SYSTEM = (
    "Tu es un data scientist expert. Tu choisis les modèles à comparer pour un dataset "
    "tabulaire. Le modèle linéaire (logistic_regression ou ridge) est une baseline rapide "
    "et interprétable, efficace sur peu de données. random_forest et extra_trees sont "
    "robustes et demandent peu de réglages. xgboost, lightgbm et hist_gradient_boosting "
    "sont performants sur des volumes moyens à grands mais peuvent surapprendre sur peu "
    "de lignes. svm est efficace sur peu de données mais lent sur de gros volumes. knn est "
    "simple mais sensible au bruit. Choisis 2 à 4 modèles parmi ceux proposés, en incluant "
    "toujours le modèle linéaire comme baseline. Justifie en 2-3 phrases. "
    "Réponds uniquement en JSON."
)


def valid_models(task: str) -> list[str]:
    """Modèles essayés par défaut."""
    return CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS


def all_models(task: str) -> list[str]:
    return valid_models(task) + EXTRA_MODELS


def baseline(task: str) -> str:
    return "logistic_regression" if task == "classification" else "ridge"


def eligible_models(
    p: DataProfile, plan: DataAnalysisPlan
) -> tuple[list[str], dict[str, str]]:
    """Modèles autorisés pour ce jeu de données, et raison d'exclusion des autres."""
    task = p.target.task
    multiclass = task == "classification" and (p.target.n_classes or 2) > 2
    excluded: dict[str, str] = {}

    if p.n_rows > SVM_MAX_ROWS:
        excluded["svm"] = f"trop lent au-delà de {SVM_MAX_ROWS} lignes ({p.n_rows} lignes)"
    if multiclass and plan.primary_metric == "roc_auc":
        excluded["svm"] = "roc_auc multiclasse exige des probabilités, indisponibles pour le SVM"
    if task == "classification" and plan.imbalance_strategy == "class_weight":
        excluded["knn"] = "ne supporte pas class_weight (la stratégie choisie serait ignorée)"

    pool = [m for m in all_models(task) if m not in excluded]
    return pool, excluded


def summarize(p: DataProfile, plan: DataAnalysisPlan, pool: list[str]) -> str:
    lines = [
        f"Tâche: {p.target.task}",
        f"Lignes: {p.n_rows}, features après nettoyage: {p.n_cols - len(plan.columns_to_drop)}",
        f"Métrique principale: {plan.primary_metric}",
        f"Stratégie de rééquilibrage: {plan.imbalance_strategy}",
    ]
    if p.target.task == "classification":
        minority = int(p.n_rows * min(p.target.class_distribution.values()))
        lines.append(f"Exemples de la classe minoritaire: {minority}")
    lines.append(f"Modèles disponibles: {pool}")
    return "\n".join(lines)


def default_ml_plan(p: DataProfile, pool: list[str]) -> MLPlan:
    models = [m for m in valid_models(p.target.task) if m in pool]
    return MLPlan(
        models=models,
        reasoning="Plan par défaut : comparer les modèles de base disponibles.",
    )


def enforce_ml_rules(
    plan: MLPlan, p: DataProfile, pool: list[str], excluded: dict[str, str]
) -> MLPlan:
    corrections: list[str] = []
    task = p.target.task

    models: list[str] = []
    for m in plan.models:
        if m in excluded:
            corrections.append(f"'{m}' retiré ({excluded[m]})")
        elif m not in pool:
            corrections.append(f"'{m}' retiré (invalide pour une tâche de {task})")
        elif m in models:
            corrections.append(f"'{m}' en double, retiré")
        else:
            models.append(m)

    base = baseline(task)
    if base not in models:
        models.insert(0, base)
        corrections.append(f"baseline '{base}' ajoutée (toujours requise)")

    for m in pool:
        if len(models) >= 2:
            break
        if m not in models:
            models.append(m)
            corrections.append(f"'{m}' ajouté (minimum 2 modèles)")

    mentioned = [m for m in all_models(task) if m in plan.reasoning and m not in models]
    if mentioned:
        corrections.append(
            f"incohérence : le raisonnement cite {mentioned}, absents de la liste retenue"
        )

    return plan.model_copy(update={"models": models, "corrections": corrections})


class MLAgent:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def plan(self, profile: DataProfile, data_plan: DataAnalysisPlan) -> MLPlan:
        pool, excluded = eligible_models(profile, data_plan)
        prompt = "Contexte du dataset :\n" + summarize(profile, data_plan, pool)
        try:
            ml_plan = self.llm.structured(prompt, MLPlan, system=SYSTEM)
        except ValueError:
            ml_plan = default_ml_plan(profile, pool)
        return enforce_ml_rules(ml_plan, profile, pool, excluded)

    def run(
        self,
        df: pd.DataFrame,
        target: str,
        profile: DataProfile,
        data_plan: DataAnalysisPlan,
    ) -> tuple[MLPlan, ExperimentResults]:
        ml_plan = self.plan(profile, data_plan)
        results = run_experiments(
            df, target, profile.target.task, data_plan, ml_plan.models
        )
        return ml_plan, results