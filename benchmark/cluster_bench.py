import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    make_blobs,
    make_moons,
)
from sklearn.metrics import adjusted_rand_score

from tools.clustering import prepare_matrix, run_clustering


def _clean(names):
    return [n.replace(" ", "_").replace("(", "").replace(")", "") for n in names]


def iris():
    d = load_iris()
    return pd.DataFrame(d.data, columns=_clean(d.feature_names)), d.target


def wine():
    d = load_wine()
    return pd.DataFrame(d.data, columns=d.feature_names), d.target


def cancer():
    d = load_breast_cancer()
    return pd.DataFrame(d.data, columns=_clean(d.feature_names)), d.target


def blobs4():
    X, y = make_blobs(n_samples=1500, centers=4, n_features=5, random_state=7)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(5)]), y


def blobs_overlap():
    X, y = make_blobs(n_samples=1500, centers=4, n_features=5, cluster_std=3.0, random_state=7)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(5)]), y


def moons():
    X, y = make_moons(n_samples=1500, noise=0.08, random_state=0)
    return pd.DataFrame(X, columns=["x0", "x1"]), y


def noise():
    X = np.random.default_rng(0).uniform(size=(1500, 5))
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(5)]), None


def gauss_corr():
    cov = 0.9 * np.ones((5, 5)) + 0.1 * np.eye(5)
    X = np.random.default_rng(0).multivariate_normal(np.zeros(5), cov, size=1500)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(5)]), None


# k_true = None : aucune structure attendue
CASES = {
    "iris": (iris, 3),
    "wine": (wine, 3),
    "cancer": (cancer, 2),
    "blobs4": (blobs4, 4),
    "blobs_overlap": (blobs_overlap, 4),
    "moons": (moons, 2),
    "noise": (noise, None),
    "gauss_corr": (gauss_corr, None),
}


def main(seeds: int = 3):
    rows = []
    for name, (load, k_true) in CASES.items():
        df, y = load()
        ari_oracle = np.nan
        if k_true is not None:
            Z, _ = prepare_matrix(df)
            lab = KMeans(n_clusters=k_true, n_init=10, random_state=0).fit_predict(Z)
            ari_oracle = adjusted_rand_score(y, lab)

        for seed in range(seeds):
            t = time.time()
            r = run_clustering(df, seed=seed)
            ari = np.nan
            if r.has_structure and y is not None:
                ari = adjusted_rand_score(y[np.array(r.row_index)], r.labels)
            rows.append(
                dict(
                    cas=name, graine=seed, k_vrai=k_true, structure=r.has_structure,
                    n_segments=r.n_clusters if r.has_structure else 0,
                    ari=round(ari, 3), ari_oracle=round(ari_oracle, 3),
                    silhouette=r.best.silhouette, nulle=r.null_silhouette,
                    stabilite=r.best.stability_ari, affinements=len(r.refinements),
                    secondes=round(time.time() - t, 1),
                )
            )
            print(f"{name:<14} graine={seed}  structure={r.has_structure}  segments={rows[-1]['n_segments']}  ARI={ari:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv("benchmark/cluster_results.csv", index=False)
    print("\nRésumé par cas (3 graines) :")
    summary = res.groupby("cas", sort=False).agg(
        k_vrai=("k_vrai", "first"),
        structure_detectee=("structure", "mean"),
        segments_median=("n_segments", "median"),
        ari_moyen=("ari", "mean"),
        ari_oracle=("ari_oracle", "first"),
        silhouette=("silhouette", "mean"),
        nulle=("nulle", "mean"),
    ).round(3)
    print(summary.to_string())


if __name__ == "__main__":
    main()