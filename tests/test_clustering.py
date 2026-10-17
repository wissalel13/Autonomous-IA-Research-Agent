import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from tools.clustering import run_clustering


def report(title: str, df: pd.DataFrame):
    t = time.time()
    r = run_clustering(df)
    print(f"\n=== {title} ({r.n_rows_used} lignes, {time.time() - t:.0f}s) ===")
    print(f"colonnes écartées : {r.dropped_columns}")
    for c in sorted(r.candidates, key=lambda c: -c.silhouette)[:5]:
        print(f"  {c.algorithm:<14} k={c.n_clusters}  sil={c.silhouette:.3f}  stab={c.stability_ari:.2f}  DB={c.davies_bouldin:.2f}")
    b = r.best
    print(f"retenu : {b.algorithm} k={b.n_clusters} | silhouette nulle={r.null_silhouette:.3f} | structure={r.has_structure}")
    print("tailles :", np.bincount(r.labels).tolist())
    print(f"segments finaux : {r.n_clusters} | affinements : {r.refinements}")


if __name__ == "__main__":
    X, _ = make_blobs(n_samples=1500, centers=4, n_features=5, random_state=7)
    blobs = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    report("Blobs (4 groupes)", blobs)

    rng = np.random.default_rng(0)
    noise = pd.DataFrame(rng.uniform(size=(1500, 5)), columns=[f"x{i}" for i in range(5)])
    report("Bruit uniforme (aucune structure)", noise)

    mixed = blobs.copy()
    mixed["segment"] = np.where(blobs["x0"] > blobs["x0"].median(), "A", "B")
    mixed["id"] = [f"id{i}" for i in range(len(mixed))]
    mixed["const"] = 1
    mixed.loc[mixed.sample(frac=0.05, random_state=1).index, "x1"] = np.nan
    report("Mixte (catégorielle, identifiant, constante, manquants)", mixed)