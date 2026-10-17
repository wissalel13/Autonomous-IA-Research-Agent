import pandas as pd

from agents.ml_agent import eligible_models
from llm.client import LLMClient
from schemas.data import DataProfile
from schemas.plan import DataAnalysisPlan, NextAction
from schemas.results import ModelResult
from tools.modeling import LOWER_IS_BETTER, run_experiments
from tools.tuning import tune_model

COMPLEXITY = {
    "logistic_regression": 0, "ridge": 0,
    "knn": 1, "random_forest": 1, "extra_trees": 1,
    "svm": 2, "hist_gradient_boosting": 2, "lightgbm": 2,
    "xgboost": 3,
}

SYSTEM = (
    "Tu es un data scientist expert qui décide de la prochaine étape d'une expérimentation. "
    "Actions possibles : 'optimize' (optimiser les hyperparamètres du modèle retenu, utile "
    "si le modèle a du potentiel et n'a pas encore été optimisé), 'add_models' (essayer des "
    "modèles non testés), 'stop' (arrêter si les résultats sont satisfaisants ou si les "
    "modèles sont à égalité). Si tu choisis 'optimize', target_model doit être le modèle "
    "retenu. Justifie en 2-3 phrases. Réponds uniquement en JSON."
)


def base_name(name: str) -> str:
    return name.removesuffix("_tuned")


def select_final(results: list[ModelResult], metric: str) -> ModelResult:
    """Modèle le plus simple dont le score est à moins d'un écart-type du meilleur."""
    lower = metric in LOWER_IS_BETTER
    signed = lambda r: -r.cv_mean if lower else r.cv_mean
    best = max(results, key=signed)
    ok = [r for r in results if signed(r) >= signed(best) - best.cv_std]
    return min(ok, key=lambda r: (COMPLEXITY[base_name(r.model_name)], -signed(r)))


def enforce_action(
    d: NextAction, final_base: str, tuned: set[str], untried: list[str]
) -> NextAction:
    corr: list[str] = []
    action, target = d.action, d.target_model

    if action == "add_models" and not untried:
        corr.append("'add_models' impossible (tous les modèles ont été essayés) -> 'stop'")
        action = "stop"

    if action == "optimize":
        if final_base in tuned:
            corr.append(f"'{final_base}' déjà optimisé -> 'stop'")
            action, target = "stop", None
        elif target != final_base:
            corr.append(
                f"cible '{target}' remplacée par '{final_base}' "
                "(modèle le plus simple dans l'écart-type du meilleur)"
            )
            target = final_base
    else:
        target = None

    reasoning = d.reasoning
    if action != d.action:
        reasoning = f"[Action corrigée par les règles : {d.action} -> {action}] " + reasoning
    return d.model_copy(
        update={
            "action": action,
            "target_model": target,
            "reasoning": reasoning,
            "corrections": corr,
        }
    )


def decide(
    llm: LLMClient, results: list[ModelResult], task: str, metric: str, pool: list[str]
) -> NextAction:
    final = select_final(results, metric)
    final_base = base_name(final.model_name)
    tried = {base_name(r.model_name) for r in results}
    tuned = {base_name(r.model_name) for r in results if r.model_name.endswith("_tuned")}
    untried = [m for m in pool if m not in tried]

    lines = [f"Métrique: {metric}", "Résultats (CV moyenne ± écart-type, test) :"]
    for r in results:
        lines.append(f"- {r.model_name}: {r.cv_mean} ± {r.cv_std}, test {r.test_score}")
    lines += [
        f"Modèle retenu (le plus simple dans l'écart-type du meilleur): {final_base}",
        f"Modèles déjà optimisés: {sorted(tuned) or 'aucun'}",
        f"Modèles non testés: {untried or 'aucun'}",
    ]
    try:
        d = llm.structured("\n".join(lines), NextAction, system=SYSTEM)
    except ValueError:
        d = NextAction(
            action="stop" if final_base in tuned else "optimize",
            target_model=final_base,
            reasoning="Décision par défaut (règles déterministes).",
        )
    return enforce_action(d, final_base, tuned, untried)


def run_agent_loop(
    df: pd.DataFrame,
    target: str,
    profile: DataProfile,
    data_plan: DataAnalysisPlan,
    initial_results: list[ModelResult],
    llm: LLMClient | None = None,
    max_steps: int = 2,
    n_trials: int = 15,
) -> tuple[ModelResult, list[ModelResult], list[NextAction]]:
    llm = llm or LLMClient()
    pool, _ = eligible_models(profile, data_plan)
    task = profile.target.task
    metric = data_plan.primary_metric
    results = list(initial_results)
    trace: list[NextAction] = []

    for _ in range(max_steps):
        d = decide(llm, results, task, metric, pool)
        trace.append(d)
        if d.action == "stop":
            break
        if d.action == "optimize":
            results.append(tune_model(df, target, task, data_plan, d.target_model, n_trials))
        elif d.action == "add_models":
            tried = {base_name(r.model_name) for r in results}
            untried = [m for m in pool if m not in tried]
            results.extend(run_experiments(df, target, task, data_plan, untried).results)

    return select_final(results, metric), results, trace