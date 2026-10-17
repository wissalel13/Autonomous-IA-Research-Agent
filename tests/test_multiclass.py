import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from agents.orchestrator import build_graph
from agents.reporting import ReportingAgent

if __name__ == "__main__":
    X, y = make_classification(
        n_samples=2000, n_features=8, n_informative=5, n_classes=3,
        weights=[0.7, 0.2, 0.1], random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["target"] = y
    df["const"] = 1
    df.loc[df.sample(frac=0.1, random_state=0).index, "f1"] = np.nan

    t = time.time()
    state = build_graph().invoke({"df": df, "target": "target"})

    p = state["profile"]
    print("Classes :", p.target.class_distribution, "ratio", p.target.imbalance_ratio)
    plan = state["data_plan"]
    print("Plan :", plan.imbalance_strategy, "| métrique", plan.primary_metric)
    print("Corrections :", plan.corrections)
    print("\nRésultats :")
    for r in state["results"]:
        print(f"  {r.model_name:<26} cv={r.cv_mean:.3f} ± {r.cv_std:.3f}  test={r.test_score:.3f}")
    print(f"\nModèle final : {state['final'].model_name}")
    print("SHAP :", [(f.feature, f.mean_abs_shap, f.direction) for f in state["explanation"].top_features[:3]])

    report = ReportingAgent().run(state)
    print("\n" + report.split("## Données")[0])
    print([l for l in report.splitlines() if "multiclasse" in l])
    print(f"\nTemps total : {time.time() - t:.0f}s")