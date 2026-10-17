import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

import tools.clustering as tc


def mixed_df():
    X, _ = make_blobs(n_samples=1500, centers=4, n_features=5, random_state=7)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    df["segment"] = np.where(df["x0"] > df["x0"].median(), "A", "B")
    return df


def customers_df():
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

    parts = [
        seg(500, (25, 4), (25000, 5000), 30, 0.9),
        seg(500, (42, 6), (80000, 12000), 12, 0.2),
        seg(500, (63, 6), (45000, 8000), 5, 0.4),
    ]
    df = pd.concat(parts, ignore_index=True)
    df["true"] = np.repeat([0, 1, 2], 500)
    df["bruit"] = rng.normal(size=len(df))
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    return df.drop(columns=["true"]), df["true"]


if __name__ == "__main__":
    mixed = mixed_df()
    customers, truth = customers_df()
    print("poids | Mixte : segments / subdivisions écartées | Clients : segments / ARI")
    for w in (1.0, 0.7, 0.5, 0.3):
        tc.CAT_WEIGHT = w
        rm = tc.run_clustering(mixed)
        rc = tc.run_clustering(customers)
        ari = adjusted_rand_score(truth.loc[rc.row_index], rc.labels)
        print(f"{w:>4}  | {rm.n_clusters} / {len(rm.refinements)}"
              f"{'':<32}| {rc.n_clusters} / {ari:.3f}")