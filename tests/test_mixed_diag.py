import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from tools.clustering import run_clustering

X, _ = make_blobs(n_samples=1500, centers=4, n_features=5, random_state=7)
blobs = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
mixed = blobs.copy()
mixed["segment"] = np.where(blobs["x0"] > blobs["x0"].median(), "A", "B")
mixed["id"] = [f"id{i}" for i in range(len(mixed))]
mixed["const"] = 1

with_nan = mixed.copy()
with_nan.loc[with_nan.sample(frac=0.05, random_state=1).index, "x1"] = np.nan

for title, df in [("sans manquants", mixed), ("avec manquants", with_nan)]:
    r = run_clustering(df)
    sizes = np.bincount(r.labels)
    print(f"\n{title} : tailles {sizes.tolist()} | affinements : {len(r.refinements)}")
    labels = pd.Series(r.labels, index=r.row_index)
    for k, n in enumerate(sizes):
        if n < 100:
            members = labels[labels == k].index
            share = df.loc[members, "x1"].isna().mean()
            print(f"  segment {k} ({n} lignes) : {share:.0%} de valeurs x1 manquantes")