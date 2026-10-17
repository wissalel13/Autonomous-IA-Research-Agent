import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from agents.data_analysis import DataAnalysisAgent
from agents.ml_agent import MLAgent
from agents.optimizer import run_agent_loop
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

    t = time.time()
    profile = profile_dataset(df, target="target")
    data_plan = DataAnalysisAgent().run(profile)
    ml_plan, res = MLAgent().run(df, "target", profile, data_plan)

    final, results, trace = run_agent_loop(df, "target", profile, data_plan, res.results)

    print("\n=== Décisions de l'agent ===")
    for i, d in enumerate(trace, 1):
        print(f"Étape {i}: {d.action} ({d.target_model})")
        print(f"  raisonnement: {d.reasoning}")
        print(f"  corrections : {d.corrections}")

    print("\n=== Résultats ===")
    for r in results:
        print(f"{r.model_name:<25} cv={r.cv_mean:.3f} ± {r.cv_std:.3f}  test={r.test_score:.3f}")
    print(f"\nModèle final : {final.model_name}")
    print(f"Temps total : {time.time() - t:.0f}s")