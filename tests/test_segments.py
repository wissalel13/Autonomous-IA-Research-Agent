import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from tools.clustering import run_clustering
from tools.segments import plot_clusters, profile_segments

rng = np.random.default_rng(3)


def seg(n, age, income, purchases, p_web):
    return pd.DataFrame(
        {
            "age": rng.normal(*age, n),
            "revenu": rng.normal(*income, n),
            "achats_par_an": rng.poisson(purchases, n),
            "canal": np.where(rng.random(n) < p_web, "web", "magasin"),
        }
    )


if __name__ == "__main__":
    parts = [
        seg(500, (25, 4), (25000, 5000), 30, 0.9),
        seg(500, (42, 6), (80000, 12000), 12, 0.2),
        seg(500, (63, 6), (45000, 8000), 5, 0.4),
    ]
    df = pd.concat(parts, ignore_index=True)
    df["true"] = np.repeat([0, 1, 2], 500)
    df["bruit"] = rng.normal(size=len(df))
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    truth = df.pop("true")

    res = run_clustering(df)
    print("Candidats par silhouette :")
    for c in sorted(res.candidates, key=lambda c: -c.silhouette)[:8]:
        print(f"  {c.algorithm:<14} k={c.n_clusters}  sil={c.silhouette:.3f}  stab={c.stability_ari:.2f}  DB={c.davies_bouldin:.2f}")
    sans_bruit = run_clustering(df.drop(columns=["bruit"]))
    print(f"Sans la colonne de bruit : k={sans_bruit.best.n_clusters}\n")
    b = res.best
    print(f"retenu : {b.algorithm} k={b.n_clusters} | structure={res.has_structure}")
    print(f"segments finaux : {res.n_clusters} | affinements : {res.refinements}")
    print(f"ARI contre les vrais segments : {adjusted_rand_score(truth.loc[res.row_index], res.labels):.3f}\n")

    for p in profile_segments(df, res):
        print(f"Segment {p.segment_id} ({p.size} clients, {p.share:.0%}) : {p.summary}")
        for f in p.features:
            print(f"    {f.feature:<14} effet={f.effect:+.2f}  {f.detail}")

    print("\ngraphique :", plot_clusters(df, res))