from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from schemas.clustering import ClusteringResult, SegmentFeature, SegmentProfile
from tools.clustering import prepare_matrix

NOTABLE = 0.5  # écart standardisé minimal pour qualifier une variable (convention « effet moyen »)


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def profile_segments(df: pd.DataFrame, result: ClusteringResult, top_k: int = 4) -> list[SegmentProfile]:
    """Décrit chaque segment par ses écarts standardisés. L'index de df doit être unique."""
    data = df.loc[result.row_index]
    labels = np.array(result.labels)
    cols = [c for c in df.columns if c not in result.dropped_columns]
    num = [c for c in cols if _is_numeric(df[c])]
    cat = [c for c in cols if c not in num]

    g_mean, g_std = data[num].mean(), data[num].std()
    profiles = []
    for k in sorted(set(labels)):
        sub = data[labels == k]
        feats: list[SegmentFeature] = []

        for c in num:
            if not g_std[c] or pd.isna(g_std[c]):
                continue
            z = (sub[c].mean() - g_mean[c]) / g_std[c]
            feats.append(
                SegmentFeature(
                    feature=c, kind="numérique", effect=round(float(z), 2),
                    detail=f"moyenne {sub[c].mean():.2f} contre {g_mean[c]:.2f} globalement",
                )
            )

        for c in cat:
            best = None
            for v in sub[c].dropna().unique():
                p_in, p_g = (sub[c] == v).mean(), (data[c] == v).mean()
                if p_g in (0.0, 1.0):
                    continue
                e = (p_in - p_g) / np.sqrt(p_g * (1 - p_g))
                if best is None or abs(e) > abs(best[0]):
                    best = (e, v, p_in, p_g)
            if best:
                e, v, p_in, p_g = best
                feats.append(
                    SegmentFeature(
                        feature=c, kind="catégorielle", effect=round(float(e), 2),
                        detail=f"'{v}' : {p_in:.0%} contre {p_g:.0%} globalement",
                    )
                )

        feats.sort(key=lambda f: -abs(f.effect))
        feats = feats[:top_k]

        words = []
        for f in feats:
            if abs(f.effect) < NOTABLE:
                continue
            if f.kind == "numérique":
                words.append(f"{f.feature} {'élevé' if f.effect > 0 else 'faible'}")
            elif f.effect > 0:
                words.append(f"{f.feature} surtout {f.detail.split(' : ')[0]}")
        summary = ", ".join(words[:3]) or "proche de la moyenne sur toutes les variables"

        profiles.append(
            SegmentProfile(
                segment_id=int(k), size=int(len(sub)), share=round(len(sub) / len(data), 3),
                features=feats, summary=summary,
            )
        )
    return profiles


def plot_clusters(df: pd.DataFrame, result: ClusteringResult, path: str = "outputs/clusters.png",
    titles: dict[int, str] | None = None,) -> str:

    data = df.loc[result.row_index]
    X, _ = prepare_matrix(data)
    pca = PCA(n_components=2, random_state=0)
    P = pca.fit_transform(X)
    labels = np.array(result.labels)

    Path(path).parent.mkdir(exist_ok=True)
    plt.figure(figsize=(6.5, 5))
    for k in sorted(set(labels)):
        m = labels == k
        plt.scatter(P[m, 0], P[m, 1], s=8, alpha=0.6, label=f"Segment {k}")
    var = pca.explained_variance_ratio_
    plt.xlabel(f"Composante 1 ({var[0]:.0%} de la variance)")
    plt.ylabel(f"Composante 2 ({var[1]:.0%} de la variance)")
    plt.title(f"Segments en projection 2D ({var.sum():.0%} de la variance conservée)")
    for k in sorted(set(labels)):
        m = labels == k
        label = (titles or {}).get(int(k), f"Segment {k}")
        plt.scatter(P[m, 0], P[m, 1], s=8, alpha=0.6, label=label)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path