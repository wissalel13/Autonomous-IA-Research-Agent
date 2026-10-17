import re

from llm.client import LLMClient
from schemas.data import DataProfile
from schemas.plan import DataAnalysisPlan
from schemas.research import Finding, ResearchReport, SentenceChoice
from tools.rag import search

ANAPHORA = re.compile(r"^(Elle|Elles|Il|Ils|On|Cela|Ceci|Ce|Cette|Ces|Alors)\b")

SYSTEM = (
    "Tu relies une décision d'analyse de données à un passage de documentation. "
    "On te donne des phrases numérotées. Choisis le numéro de LA phrase qui justifie "
    'le mieux la décision. Réponds uniquement en JSON : {"index": numéro}.'
)

STRATEGY_QUERY = {
    "class_weight": "pondération des classes class_weight",
    "smote": "SMOTE exemples synthétiques",
    "undersampling": "sous-échantillonnage",
    "none": "classes déséquilibrées problème",
}


def build_topics(
    profile: DataProfile, plan: DataAnalysisPlan, final_model: str, explained: bool
) -> list[dict]:
    topics = []
    is_clf = profile.target.task == "classification"

    if is_clf and (profile.target.imbalance_ratio or 1) >= 4:
        topics.append(
            {
                "topic": "Rééquilibrage des classes",
                "decision": f"stratégie de rééquilibrage : {plan.imbalance_strategy}",
                "query": STRATEGY_QUERY[plan.imbalance_strategy],
                "sources": ["desequilibre"],
            }
        )
    topics.append(
        {
            "topic": "Choix de la métrique",
            "decision": f"métrique principale : {plan.primary_metric}",
            "query": f"{plan.primary_metric} métrique",
            "sources": ["metriques"],
        }
    )
    if any(c.missing_pct > 0 for c in profile.columns):
        topics.append(
            {
                "topic": "Valeurs manquantes",
                "decision": f"imputation numérique : {plan.numeric_imputation}",
                "query": "imputation numérique médiane moyenne",
                "sources": ["valeurs_manquantes"],
            }
        )

    dropped = set(plan.columns_to_drop)
    const_dropped = [c for c in profile.constant_columns if c in dropped]
    pair_cols = {c for a, b, _ in profile.highly_correlated_pairs for c in (a, b)}
    dup_dropped = sorted((dropped & pair_cols) - set(const_dropped))
    if const_dropped:
        topics.append(
            {
                "topic": "Colonnes constantes",
                "decision": f"suppression de : {', '.join(const_dropped)}",
                "query": "colonne constante aucune information supprimer",
                "sources": ["variables_correlees"],
            }
        )
    if dup_dropped:
        topics.append(
            {
                "topic": "Colonnes redondantes",
                "decision": f"suppression de : {', '.join(dup_dropped)}",
                "query": "variables fortement corrélées redondantes en garder une",
                "sources": ["variables_correlees"],
            }
        )

    topics.append(
        {
            "topic": "Choix du modèle",
            "decision": f"modèle retenu : {final_model}",
            "query": "baseline modèle simple comparer écart-type",
            "sources": ["choix_modeles"],
        }
    )
    topics.append(
        {
            "topic": "Protocole d'évaluation",
            "decision": "validation croisée stratifiée dans un pipeline",
            "query": "validation croisée pipeline fuite de données stratifié",
            "sources": ["validation_croisee"],
        }
    )
    if explained:
        topics.append(
            {
                "topic": "Explicabilité (limite à connaître)",
                "decision": "importance des variables par SHAP",
                "query": (
                    "SHAP variables corrélées crédit réparti"
                    if any(
                        a not in plan.columns_to_drop and b not in plan.columns_to_drop
                        for a, b, _ in profile.highly_correlated_pairs
                    )
                    else "importance globale moyenne valeurs absolues SHAP"
                ),
                "sources": ["explicabilite_shap"],
            }
        )
    return topics


def split_sentences(passage: str, section: str) -> list[str]:
    marker = f"{section} : "
    body = passage.split(marker, 1)[-1].strip() if marker in passage else passage.strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]


class ResearchAgent:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def _pick(self, decision: str, passage: str, section: str) -> tuple[str, str, str | None]:
        """Retourne (phrase, origine, correction éventuelle)."""
        sentences = split_sentences(passage, section)
        if len(sentences) == 1:
            return sentences[0], "single", None

        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
        prompt = f"Décision : {decision}\nPhrases du passage :\n{numbered}"
        try:
            idx = self.llm.structured(prompt, SentenceChoice, system=SYSTEM).index
        except ValueError:
            return sentences[0], "default", "sortie LLM invalide -> première phrase"
        if not 1 <= idx <= len(sentences):
            return sentences[0], "default", f"numéro {idx} hors limites -> première phrase"
        chosen = sentences[idx - 1]
        if idx > 1 and ANAPHORA.match(chosen):
            chosen = " ".join(sentences[max(0, idx - 3): idx])
        return chosen, "llm", None

    def run(
        self,
        profile: DataProfile,
        plan: DataAnalysisPlan,
        final_model: str,
        explained: bool = True,
    ) -> ResearchReport:
        findings: list[Finding] = []
        corrections: list[str] = []

        for t in build_topics(profile, plan, final_model, explained):
            hits = search(t["query"], k=1, sources=t["sources"])
            if not hits:
                corrections.append(f"aucun passage trouvé pour '{t['topic']}'")
                continue
            hit = hits[0]
            note, origin, corr = self._pick(t["decision"], hit["text"], hit["section"])
            if corr:
                corrections.append(f"{t['topic']} : {corr}")
            findings.append(
                Finding(
                    topic=t["topic"],
                    decision=t["decision"],
                    passage_id=hit["id"],
                    section=hit["section"],
                    passage_text=hit["text"],
                    note=note,
                    note_source=origin,
                )
            )
        return ResearchReport(findings=findings, corrections=corrections)