import re
from datetime import date
from pathlib import Path

from llm.client import LLMClient
from schemas.report import SummaryText
from tools.modeling import LOWER_IS_BETTER

SYSTEM = (
    "Tu rédiges le résumé d'un rapport d'analyse de données, en français, en 2 ou 3 "
    "phrases. Utilise uniquement les faits fournis. Recopie les chiffres et le nom de la "
    "métrique exactement comme ils sont écrits (par exemple average_precision), sans les "
    "traduire ni les reformuler. Réponds uniquement en JSON : "
    '{"text": "..."}.'
)

FOREIGN = re.compile(r"\b(the|and|with|para|não|porque|pero|mejor)\b", re.I)


def _numbers(text: str) -> set[float]:
    return {float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", text)}


def build_facts(s: dict) -> str:
    p, plan, final = s["profile"], s["data_plan"], s["final"]
    lines = [
        f"Jeu de données : {p.n_rows} lignes, {p.n_cols} variables, "
        f"cible '{p.target.name}', tâche de {p.target.task}.",
    ]
    if p.target.task == "classification":
        lines.append(f"Ratio de déséquilibre des classes : {p.target.imbalance_ratio}.")
    lines += [
        f"Métrique principale : {plan.primary_metric}.",
        f"Stratégie de rééquilibrage : {plan.imbalance_strategy}.",
        f"Modèle final : {final.model_name}, score de validation croisée "
        f"{final.cv_mean} ± {final.cv_std}, score de test {final.test_score}.",
        "Variables les plus influentes : "
        + ", ".join(f.feature for f in s["explanation"].top_features[:3])
        + ".",
    ]
    return "\n".join(lines)


def default_summary(s: dict) -> str:
    p, plan, final = s["profile"], s["data_plan"], s["final"]
    top = ", ".join(f.feature for f in s["explanation"].top_features[:3])
    return (
        f"Analyse d'un jeu de {p.n_rows} lignes (tâche de {p.target.task}). "
        f"Le modèle retenu est {final.model_name} : {plan.primary_metric} "
        f"{final.cv_mean} ± {final.cv_std} en validation croisée, {final.test_score} sur le test. "
        f"Les variables les plus influentes sont {top}."
    )


def build_caveats(s: dict) -> list[str]:
    p, plan, results = s["profile"], s["data_plan"], s["results"]
    metric = plan.primary_metric
    sign = -1 if metric in LOWER_IS_BETTER else 1
    cav: list[str] = []

    if p.target.task == "classification":
        share = min(p.target.class_distribution.values())
        minority = int(p.n_rows * share)
        if minority < 300:
            cav.append(
                f"Seulement ~{minority} exemples de la classe minoritaire "
                f"(environ {int(minority * 0.2)} dans le jeu de test) : le score de test est "
                "très variable, il faut privilégier la validation croisée."
            )

    best = max(results, key=lambda r: sign * r.cv_mean)
    close = [
        r.model_name
        for r in results
        if r is not best and sign * r.cv_mean >= sign * best.cv_mean - best.cv_std
    ]
    if close:
        cav.append(
            f"Plusieurs modèles sont à moins d'un écart-type du meilleur ({best.model_name}) : "
            f"{', '.join(close)}. Leurs performances sont statistiquement indiscernables, "
            "le plus simple est retenu."
        )

    tuned = [r for r in results if r.model_name.endswith("_tuned")]
    if tuned:
        cav.append(
            "Les scores des modèles optimisés sont légèrement optimistes : les "
            "hyperparamètres ont été choisis sur les mêmes folds de validation croisée."
        )
        for t in tuned:
            base = next((r for r in results if r.model_name == t.model_name.removesuffix("_tuned")), None)
            if base and abs(t.cv_mean - base.cv_mean) < t.cv_std:
                cav.append(
                    f"Le gain de l'optimisation de {base.model_name} "
                    f"({base.cv_mean} -> {t.cv_mean}) est inférieur à l'écart-type : négligeable."
                )
    if p.target.task == "classification" and (p.target.n_classes or 2) > 2:
        meaning = {
            "f1": "le score f1 est une moyenne macro (chaque classe pèse autant)",
            "roc_auc": "le roc_auc est calculé en un contre tous",
            "balanced_accuracy": "la balanced accuracy est la moyenne du rappel de chaque classe (chaque classe pèse autant)",
            "accuracy": "l'accuracy favorise les classes fréquentes",
        }
        cav.append(
            f"Problème multiclasse ({p.target.n_classes} classes) : "
            f"{meaning.get(metric, 'métrique adaptée au multiclasse')}."
        )

    remaining = [
        (a, b, v)
        for a, b, v in p.highly_correlated_pairs
        if a not in plan.columns_to_drop and b not in plan.columns_to_drop
    ]
    if remaining:
        cav.append(
            "Des variables fortement corrélées subsistent : SHAP répartit le crédit entre "
            "elles, leurs effets doivent être interprétés conjointement. SHAP décrit le "
            "modèle, pas des relations de cause à effet."
        )
    return cav
        


def render_markdown(s: dict, summary: str, summary_source: str) -> str:
    p, plan, final = s["profile"], s["data_plan"], s["final"]
    expl, research = s["explanation"], s["research"]
    metric = plan.primary_metric

    out = [f"# Rapport d'analyse automatique", f"*Généré le {date.today().isoformat()}*", ""]
    out += ["## Résumé", summary, f"*(résumé : {summary_source})*", ""]

    out += ["## Données"]
    out += [f"- {p.n_rows} lignes, {p.n_cols} variables, {p.n_duplicates} doublons"]
    out += [f"- Cible : `{p.target.name}` ({p.target.task})"]
    if p.target.task == "classification":
        out += [f"- Distribution des classes : {p.target.class_distribution} (ratio {p.target.imbalance_ratio})"]
    out += [""]

    out += ["## Décisions de l'agent"]
    out += [f"- Colonnes supprimées : {', '.join(plan.columns_to_drop) or 'aucune'}"]
    out += [f"- Imputation numérique : {plan.numeric_imputation}"]
    out += [f"- Rééquilibrage : {plan.imbalance_strategy}"]
    out += [f"- Métrique principale : {metric}"]
    out += [f"- Modèles comparés : {', '.join(s['ml_plan'].models)}", ""]

    out += ["## Résultats", f"| Modèle | CV ({metric}) | Test |", "|---|---|---|"]
    for r in s["results"]:
        star = " ★" if r.model_name == final.model_name else ""
        out += [f"| {r.model_name}{star} | {r.cv_mean} ± {r.cv_std} | {r.test_score} |"]
    out += ["", f"**Modèle retenu : `{final.model_name}`** (run MLflow `{final.run_id}`)", ""]

    out += ["## Explicabilité (SHAP)", "| Variable | Importance moyenne \\|SHAP\\| | Effet |", "|---|---|---|"]
    for f in expl.top_features:
        out += [f"| {f.feature} | {f.mean_abs_shap} | {f.direction} |"]
    out += ["", f"![SHAP]({Path(expl.plot_path).name})", ""]

    out += ["## Justifications documentées"]
    for f in research.findings:
        out += [f"**{f.topic}** — {f.decision}", f"> {f.note}", f"> *Source : `{f.passage_id}` ({f.section})*", ""]

    out += ["## Limites"]
    out += [f"- {c}" for c in build_caveats(s)]
    out += [""]

    out += ["## Traçabilité : corrections appliquées aux sorties du LLM"]
    rows = [("Plan données", c) for c in plan.corrections]
    rows += [("Plan ML", c) for c in s["ml_plan"].corrections]
    for i, d in enumerate(s["trace"], 1):
        rows += [(f"Boucle, étape {i}", c) for c in d.corrections]
    rows += [("Recherche", c) for c in research.corrections]
    if rows:
        out += ["| Étape | Correction |", "|---|---|"]
        out += [f"| {a} | {b} |" for a, b in rows]
    else:
        out += ["Aucune correction nécessaire."]
    return "\n".join(out) + "\n"


class ReportingAgent:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def _summary(self, s: dict) -> tuple[str, str]:
        facts = build_facts(s)
        try:
            text = self.llm.structured(facts, SummaryText, system=SYSTEM).text.strip()
        except ValueError:
            return default_summary(s), "par défaut (sortie LLM invalide)"

        sentences = [x for x in re.split(r"(?<=[.!?])\s+", text) if x]
        if not text or len(text) > 600 or len(sentences) > 4:
            return default_summary(s), "par défaut (longueur invalide)"
        if FOREIGN.search(text):
            return default_summary(s), "par défaut (langue incorrecte)"
        if s["data_plan"].primary_metric not in text:
            return default_summary(s), "par défaut (nom de la métrique absent)"
        invented = _numbers(text) - _numbers(facts)
        if invented:
            return default_summary(s), f"par défaut (chiffres absents des faits : {sorted(invented)})"
        return text, "généré par le LLM, validé"

    def run(self, state: dict) -> str:
        summary, source = self._summary(state)
        return render_markdown(state, summary, source)