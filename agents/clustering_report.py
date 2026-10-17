from datetime import date
from pathlib import Path

from schemas.clustering import ClusteringReport
from tools.clustering import MAX_ROWS, MIN_SUB_ROWS, MIN_SUB_SHARE, STABILITY_MIN, STRUCTURE_MARGIN


def build_clustering_caveats(rep: ClusteringReport) -> list[str]:
    r = rep.result
    cav = [
        f"Les seuils de décision sont des heuristiques non calibrées : stabilité minimale "
        f"{STABILITY_MIN}, écart minimal de silhouette {STRUCTURE_MARGIN}, poids des variables "
        f"catégorielles {r.cat_weight}, taille minimale d'un sous-segment {MIN_SUB_ROWS} lignes "
        f"ou {MIN_SUB_SHARE:.0%} de son segment. Ils ont été ajustés sur des jeux de données synthétiques.",
        "KMeans et le mélange gaussien supposent des groupes plutôt arrondis : des groupes de forme "
        "allongée ou en croissant peuvent être mal découpés.",
        "Des groupes qui se chevauchent beaucoup peuvent être fusionnés : l'outil sous-segmente "
        "plutôt qu'il ne sur-segmente.",
    ]
    if r.has_structure:
        cav.append(
            "Les titres décrivent les caractéristiques mesurées, sans interprétation métier : "
            "un segment « âge élevé » n'est pas forcément « fidèle » ou « senior » au sens commercial."
        )
        cav.append(
            "Les effets sont des écarts à la moyenne globale, pas des relations de cause à effet."
        )
    if r.refinements:
        cav.append(
            "Chaque segment testé pour une subdivision est un test de plus : le risque de "
            "faux positif augmente avec leur nombre."
        )
    if r.n_rows_used >= MAX_ROWS:
        cav.append(f"Le clustering est calculé sur un échantillon de {MAX_ROWS} lignes.")
    return cav


def render_clustering_markdown(rep: ClusteringReport) -> str:
    r, b = rep.result, rep.result.best
    out = ["# Rapport de segmentation automatique", f"*Généré le {date.today().isoformat()}*", ""]
    out += ["## Verdict", rep.verdict, ""]

    out += [
        "## Données",
        f"- {r.n_rows_used} lignes analysées",
        f"- Colonnes écartées (constantes ou identifiants) : {', '.join(r.dropped_columns) or 'aucune'}",
        "",
    ]

    out += [
        "## Choix du nombre de groupes",
        "| Algorithme | k | Silhouette | Davies-Bouldin | Stabilité |",
        "|---|---|---|---|---|",
    ]
    for c in sorted(r.candidates, key=lambda c: -c.silhouette)[:6]:
        mark = " ★" if r.has_structure and (c.algorithm, c.n_clusters) == (b.algorithm, b.n_clusters) else ""
        out += [f"| {c.algorithm}{mark} | {c.n_clusters} | {c.silhouette} | {c.davies_bouldin} | {c.stability_ari} |"]

    legend = (
        "★ = point de départ retenu. "
        if r.has_structure
        else "Aucun candidat retenu : le tableau n'est donné qu'à titre indicatif. "
    )
    out += [
        "",
        f"*{legend}Silhouette de référence pour des données sans groupes "
        f"(colonnes mélangées, et gaussienne de même covariance) : {r.null_silhouette}.*",
        "",
    ]

    if r.has_structure:
        if r.refinements:
            out += ["## Affinements"] + [f"- {x}" for x in r.refinements] + [""]

        out += ["## Segments", "| Segment | Taille | Part | Caractéristiques marquantes |", "|---|---|---|---|"]
        for s in rep.segments:
            tag = " *(titre LLM)*" if s.text_source == "llm" else ""
            out += [f"| {s.title}{tag} | {s.profile.size} | {s.profile.share:.0%} | {s.profile.summary} |"]
        out += [""]
        if rep.plot_path:
            out += [f"![Segments]({Path(rep.plot_path).name})", ""]

        out += [
            "## Détail par segment",
            "*L'effet est l'écart à la moyenne globale : en écarts-types pour une variable numérique, "
            "en écart de proportion standardisé pour une catégorielle. En dessous de 0,5 en valeur "
            "absolue, l'écart est faible.*",
            "",
        ]
        for s in rep.segments:
            out += [f"### {s.title}", f"{s.profile.size} individus ({s.profile.share:.0%})"]
            out += [f"- **{f.feature}** : effet {f.effect:+.2f}, {f.detail}" for f in s.profile.features]
            out += [""]

    out += ["## Limites"] + [f"- {c}" for c in build_clustering_caveats(rep)] + [""]

    out += ["## Traçabilité"]
    if rep.corrections:
        out += ["| Étape | Correction |", "|---|---|"] + [f"| Nommage | {c} |" for c in rep.corrections]
    else:
        out += ["Aucune correction nécessaire."]
    out += ["", f"Graine aléatoire : 42. Run MLflow : `{rep.run_id}`."]
    return "\n".join(out) + "\n"