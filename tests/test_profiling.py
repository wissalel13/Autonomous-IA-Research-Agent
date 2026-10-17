import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from tools.profiling import profile_dataset

if __name__ == "__main__":
    X, y = make_classification(
        n_samples=2000, n_features=6, weights=[0.95, 0.05], random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    df["const"] = 1
    df["f0_copy"] = df["f0"] * 2 + 0.001
    df.loc[df.sample(frac=0.2, random_state=0).index, "f1"] = np.nan

    profile = profile_dataset(df, target="target")
    print(profile.target)
    print("constantes :", profile.constant_columns)
    print("corrélées  :", profile.highly_correlated_pairs)
    print("f1 manquants (%) :", next(c.missing_pct for c in profile.columns if c.name == "f1"))