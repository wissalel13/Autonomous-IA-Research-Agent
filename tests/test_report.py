from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from agents.orchestrator import build_graph
from agents.reporting import ReportingAgent

if __name__ == "__main__":
    X, y = make_classification(
        n_samples=2000, n_features=6, weights=[0.95, 0.05], random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    df["const"] = 1
    df["f0_copy"] = df["f0"] * 2 + 0.001
    df.loc[df.sample(frac=0.2, random_state=0).index, "f1"] = np.nan

    state = build_graph().invoke({"df": df, "target": "target"})
    report = ReportingAgent().run(state)

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/report.md").write_text(report, encoding="utf-8")
    print(report)