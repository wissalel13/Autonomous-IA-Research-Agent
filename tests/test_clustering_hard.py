import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons

from tests.test_clustering import report

if __name__ == "__main__":
    # 1. groupes qui se chevauchent
    X, _ = make_blobs(n_samples=1500, centers=4, n_features=5, cluster_std=3.0, random_state=7)
    report("Blobs chevauchants", pd.DataFrame(X, columns=[f"x{i}" for i in range(5)]))

    # 2. une seule gaussienne très corrélée : aucune structure en groupes
    rng = np.random.default_rng(0)
    cov = 0.9 * np.ones((5, 5)) + 0.1 * np.eye(5)
    G = rng.multivariate_normal(np.zeros(5), cov, size=1500)
    report("Gaussienne unique corrélée (aucun groupe)", pd.DataFrame(G, columns=[f"x{i}" for i in range(5)]))

    # 3. deux lunes : groupes réels mais non sphériques
    M, _ = make_moons(n_samples=1500, noise=0.08, random_state=0)
    report("Deux lunes (non sphérique)", pd.DataFrame(M, columns=["x0", "x1"]))