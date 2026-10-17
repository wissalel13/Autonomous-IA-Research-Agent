import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from tools.clustering import run_clustering

X, true = make_blobs(n_samples=1500, centers=4, n_features=5, random_state=7)
blobs = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
mixed = blobs.copy()
mixed["segment"] = np.where(blobs["x0"] > blobs["x0"].median(), "A", "B")

for title, df in [
    ("avec la catégorielle", mixed),
    ("sans la catégorielle", mixed.drop(columns=["segment"])),
]:
    r = run_clustering(df)
    print(f"\n{title} : {r.n_clusters} segments | notes : {r.refinements}")

print("\nModalités de 'segment' dans chaque blob d'origine :")
print(pd.crosstab(true, mixed["segment"]))