import json
import re

import mlflow
import pandas as pd

from llm.client import LLMClient
from schemas.clustering import (
    ClusteringReport,
    ClusteringResult,
    NamedSegment,
    SegmentProfile,
    SegmentTitle,
)
from tools.clustering import run_clustering
from tools.segments import plot_clusters, profile_segments
from config import setup_mlflow

SYSTEM = (
    "Tu donnes un titre descriptif court à un groupe d'individus issu d'un clustering, "
    "d'après les faits fournis : 4 mots maximum, sans chiffre ni numéro. Le titre doit "
    "refléter uniquement les caractéristiques fournies. "
    'Réponds uniquement en JSON : {"title": "..."}.'
)

FOREIGN = re.compile(r"\b(the|and|with|para|não|porque|pero|mejor)\b", re.I)
GENERIC = re.compile(r"(segment|groupe|cluster)\s*\d*", re.I)


def segment_facts(p: SegmentProfile) -> str:
    lines = [f"Profil à nommer : {p.size} individus ({p.share:.0%} du total)."]
    for f in p.features:
        lines.append(f"- {f.feature} ({f.kind}) : effet {f.effect:+.2f}, {f.detail}")
    return "\n".join(lines)


def default_text(p: SegmentProfile) -> str:
    return f"Segment de {p.size} individus ({p.share:.0%}) : {p.summary}."

def default_title(p: SegmentProfile) -> str:
    if p.summary.startswith("proche de la moyenne"):
        return f"Segment {p.segment_id} (proche de la moyenne)"
    return " / ".join(p.summary.replace("_", " ").split(", ")[:2])


def build_verdict(r: ClusteringResult) -> str:
    b = r.best
    if r.has_structure:
        return (
            f"Structure détectée : {r.n_clusters} segments (clustering de départ : {b.algorithm}, "
            f"k={b.n_clusters}, silhouette {b.silhouette} contre {r.null_silhouette} pour des "
            f"données sans groupes, stabilité {b.stability_ari})."
        )
    return (
        f"Aucun groupe clair : la meilleure silhouette ({b.silhouette}) n'est pas nettement "
        f"supérieure à celle de données sans groupes ({r.null_silhouette}), ou le clustering "
        f"est instable (stabilité {b.stability_ari}). Aucun segment n'est proposé."
    )


class ClusteringAgent:
    def __init__(self, llm: LLMClient | None = None, use_llm_titles: bool = False):
        self.llm = llm or LLMClient()
        self.use_llm_titles = use_llm_titles

    def _check_title(self, title: str, p: SegmentProfile, cols: list[str], used: set[str]) -> str | None:
        """Retourne la raison du rejet, ou None si le titre est valide."""
        if not 1 <= len(title) <= 40:
            return "titre vide ou trop long"
        if len(title.split()) > 5:
            return "plus de 5 mots"
        if re.search(r"\d", title):
            return "chiffre dans le titre"
        if GENERIC.fullmatch(title):
            return "titre générique (numéro sans description)"
        if title.lower() in used:
            return "titre déjà utilisé"
        if FOREIGN.search(title):
            return "langue incorrecte"
        mine = {f.feature for f in p.features}
        others = [c for c in cols if c not in mine and re.search(rf"\b{re.escape(c)}\b", title, re.I)]
        if others:
            return f"variables absentes du profil du segment : {others}"
        return None

    def _name_segments(self, profiles: list[SegmentProfile], cols: list[str]):
        used: set[str] = set()
        segments: list[NamedSegment] = []
        corrections: list[str] = []
        for p in profiles:
            title, source = default_title(p), "rules"
            if title.lower() in used:
                title = f"{title} ({p.segment_id})"
            if self.use_llm_titles:
                try:
                    out = self.llm.structured(segment_facts(p), SegmentTitle, system=SYSTEM)
                    candidate = out.title.strip()
                    why = self._check_title(candidate, p, cols, used)
                    if why is None:
                        title, source = candidate, "llm"
                    else:
                        corrections.append(
                            f"Segment {p.segment_id} : titre LLM rejeté ({why}) | titre={candidate[:40]!r}"
                        )
                except ValueError:
                    corrections.append(f"Segment {p.segment_id} : sortie LLM invalide")
            used.add(title.lower())
            # text_source décrit l'origine du titre ; la description est toujours produite par le code
            segments.append(
                NamedSegment(profile=p, title=title, description=default_text(p), text_source=source)
            )
        return segments, corrections
    def _log(self, result: ClusteringResult, segments: list[NamedSegment],
             plot_path: str | None, experiment_name: str) -> str:
        setup_mlflow(experiment_name)
        b = result.best
        with mlflow.start_run(run_name="clustering") as run:
            mlflow.log_params(
                {
                    "start_algorithm": b.algorithm,
                    "start_k": b.n_clusters,
                    "has_structure": result.has_structure,
                    "cat_weight": result.cat_weight,
                    "n_rows_used": result.n_rows_used,
                    "dropped_columns": ",".join(result.dropped_columns) or "none",
                    "n_refinements": len(result.refinements),
                }
            )
            mlflow.log_metrics(
                {
                    "silhouette": b.silhouette,
                    "null_silhouette": result.null_silhouette,
                    "stability_ari": b.stability_ari,
                    "davies_bouldin": b.davies_bouldin,
                    "n_segments": result.n_clusters if result.has_structure else 0,
                    "titles_from_llm": sum(s.text_source == "llm" for s in segments),
                }
            )
            mlflow.log_text(
                json.dumps([s.model_dump() for s in segments], ensure_ascii=False, indent=2),
                "segments.json",
            )
            if plot_path:
                mlflow.log_artifact(plot_path, artifact_path="clusters")
            return run.info.run_id

    def run(self, df: pd.DataFrame, experiment_name: str = "ai-research-agent-clustering") -> ClusteringReport:
        df = df.reset_index(drop=True)
        result = run_clustering(df)
        corrections: list[str] = []
        segments: list[NamedSegment] = []
        plot_path = None

        if result.has_structure:
            profiles = profile_segments(df, result)
            cols = [c for c in df.columns if c not in result.dropped_columns]
            segments, corrections = self._name_segments(profiles, cols)
            plot_path = plot_clusters(
                df, result, titles={s.profile.segment_id: s.title for s in segments}
            )

        run_id = self._log(result, segments, plot_path, experiment_name)
        return ClusteringReport(
            result=result,
            verdict=build_verdict(result),
            segments=segments,
            corrections=corrections,
            plot_path=plot_path,
            run_id=run_id,
        )