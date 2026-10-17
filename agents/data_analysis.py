from llm.client import LLMClient
from schemas.data import DataProfile
from schemas.plan import (
    CLASSIFICATION_METRICS,
    MULTICLASS_METRICS,
    REGRESSION_METRICS,
    DataAnalysisPlan,
)

SYSTEM = (
    "Tu es un data scientist expert. À partir du profil d'un dataset, tu choisis "
    "les traitements de données. Règles : si les classes sont déséquilibrées "
    "(ratio >= 4), évite 'accuracy' et choisis une stratégie de rééquilibrage ; "
    "si des valeurs numériques sont asymétriques ou contiennent des extrêmes, "
    "préfère l'imputation par la médiane. Justifie en 2-3 phrases. "
    "Réponds uniquement en JSON."
)


def droppable_candidates(p: DataProfile) -> set[str]:
    cands = set(p.constant_columns) | set(p.high_cardinality_columns)
    for a, b, _ in p.highly_correlated_pairs:
        cands |= {a, b}
    cands |= {c.name for c in p.columns if c.missing_pct > 60}
    return cands


def summarize_profile(p: DataProfile) -> str:
    lines = [
        f"Lignes: {p.n_rows}, colonnes: {p.n_cols}, doublons: {p.n_duplicates}",
        f"Cible: {p.target.name} (tâche: {p.target.task})",
    ]
    if p.target.task == "classification":
        lines.append(
            f"Distribution des classes: {p.target.class_distribution}, "
            f"ratio de déséquilibre: {p.target.imbalance_ratio}"
        )
        if (p.target.n_classes or 2) > 2:
            lines.append(
                f"Problème multiclasse ({p.target.n_classes} classes) : 'average_precision' "
                "est indisponible, préfère 'f1' (moyenne macro) ou 'balanced_accuracy'."
            )
        
    missing = {c.name: c.missing_pct for c in p.columns if c.missing_pct > 0}
    lines.append(f"Colonnes avec valeurs manquantes (%): {missing or 'aucune'}")
    skewed = {c.name: c.skew for c in p.columns if c.skew is not None and abs(c.skew) > 1}
    lines.append(f"Colonnes très asymétriques (skew): {skewed or 'aucune'}")
    lines.append(f"Colonnes constantes: {p.constant_columns or 'aucune'}")
    lines.append(f"Colonnes à cardinalité très haute: {p.high_cardinality_columns or 'aucune'}")
    lines.append(f"Paires très corrélées: {p.highly_correlated_pairs or 'aucune'}")
    lines.append(
        f"Colonnes que tu peux supprimer (uniquement parmi celles-ci): "
        f"{sorted(droppable_candidates(p)) or 'aucune'}"
    )
    return "\n".join(lines)


def default_plan(p: DataProfile) -> DataAnalysisPlan:
    is_clf = p.target.task == "classification"
    multiclass = is_clf and (p.target.n_classes or 2) > 2
    imbalanced = is_clf and (p.target.imbalance_ratio or 1) >= 4
    if not is_clf:
        metric = "rmse"
    elif multiclass:
        metric = "f1"  # moyenne macro
    else:
        metric = "average_precision" if imbalanced else "f1"
    return DataAnalysisPlan(
        columns_to_drop=sorted(p.constant_columns),
        numeric_imputation="median",
        imbalance_strategy="class_weight" if imbalanced else "none",
        primary_metric=metric,
        reasoning="Plan par défaut (règles déterministes).",
    )


def enforce_rules(plan: DataAnalysisPlan, p: DataProfile) -> DataAnalysisPlan:
    """Valide et corrige la sortie du LLM, en traçant chaque correction."""
    corrections: list[str] = []
    allowed = droppable_candidates(p)

    # 1. suppression : uniquement des candidats valides
    drop: set[str] = set()
    for c in plan.columns_to_drop:
        if c in allowed:
            drop.add(c)
        else:
            corrections.append(f"'{c}' retirée de la suppression (pas un candidat valide)")

    # 2. constantes : toujours supprimées
    for c in p.constant_columns:
        if c not in drop:
            drop.add(c)
            corrections.append(f"'{c}' supprimée (colonne constante)")

    # 3. doublons quasi exacts : on garde la première colonne de chaque paire
    for a, b, v in p.highly_correlated_pairs:
        if v >= 0.999 and a not in drop and b not in drop:
            drop.add(b)
            corrections.append(f"'{b}' supprimée (corrélation {v} avec '{a}')")

    # 4. ne jamais supprimer les deux colonnes d'une paire corrélée
    for a, b, _ in p.highly_correlated_pairs:
        if a in drop and b in drop:
            drop.discard(a)
            corrections.append(f"'{a}' conservée (sa paire '{b}' est déjà supprimée)")

    is_clf = p.target.task == "classification"
    multiclass = is_clf and (p.target.n_classes or 2) > 2
    imbalanced = is_clf and (p.target.imbalance_ratio or 1) >= 4
    default = default_plan(p)

    # 5. métrique cohérente avec la tâche (et avec le nombre de classes)
    if not is_clf:
        valid = REGRESSION_METRICS
    elif multiclass:
        valid = MULTICLASS_METRICS
    else:
        valid = CLASSIFICATION_METRICS

    metric = plan.primary_metric
    if metric not in valid:
        metric = default.primary_metric
        corrections.append(f"métrique '{plan.primary_metric}' invalide pour cette tâche -> '{metric}'")
    elif imbalanced and metric == "accuracy":
        metric = default.primary_metric
        corrections.append(f"'accuracy' trompeuse avec classes déséquilibrées -> '{metric}'")
    elif (p.target.imbalance_ratio or 1) >= 10 and (
        metric == "roc_auc" or (metric == "balanced_accuracy" and not multiclass)
    ):
        metric = default.primary_metric
        corrections.append(
            f"'{plan.primary_metric}' peu informatif avec une classe positive rare "
            f"(ratio >= 10) -> '{metric}'"
        )
    # 6. rééquilibrage
    strategy = plan.imbalance_strategy
    if not imbalanced and strategy != "none":
        corrections.append(f"stratégie '{strategy}' inutile (classes équilibrées) -> 'none'")
        strategy = "none"
    if imbalanced:
        minority = int(p.n_rows * min(p.target.class_distribution.values()))
        if strategy == "undersampling" and minority < 1000:
            corrections.append(
                f"'undersampling' écarté (seulement ~{minority} exemples minoritaires) -> 'class_weight'"
            )
            strategy = "class_weight"
        if strategy == "smote" and minority < 20:
            corrections.append(
                f"'smote' écarté (seulement ~{minority} exemples minoritaires) -> 'class_weight'"
            )
            strategy = "class_weight"

    return plan.model_copy(
        update={
            "columns_to_drop": sorted(drop),
            "primary_metric": metric,
            "imbalance_strategy": strategy,
            "corrections": corrections,
        }
    )


class DataAnalysisAgent:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def run(self, profile: DataProfile) -> DataAnalysisPlan:
        prompt = "Voici le profil du dataset :\n" + summarize_profile(profile)
        try:
            plan = self.llm.structured(prompt, DataAnalysisPlan, system=SYSTEM)
        except ValueError:
            plan = default_plan(profile)
        return enforce_rules(plan, profile)