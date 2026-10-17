import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from agents.research import ResearchAgent
from schemas.plan import DataAnalysisPlan
from tools.profiling import profile_dataset
from tools.rag import build_index

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
    plan = DataAnalysisPlan(
        columns_to_drop=["const", "f0_copy"],
        numeric_imputation="median",
        imbalance_strategy="class_weight",
        primary_metric="average_precision",
        reasoning="plan de test",
    )

    build_index()
    t = time.time()
    report = ResearchAgent().run(profile, plan, final_model="logistic_regression_tuned")

    for f in report.findings:
        print(f"\n[{f.topic}] {f.decision}")
        print(f"  source : {f.passage_id} ({f.section})")
        print(f"  note ({f.note_source}) : {f.note}")
    print("\nCorrections :", report.corrections)
    print(f"Temps : {time.time() - t:.0f}s")