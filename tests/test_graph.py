import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from agents.orchestrator import build_graph

if __name__ == "__main__":
    X, y = make_classification(
        n_samples=2000, n_features=6, weights=[0.95, 0.05], random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    df["const"] = 1
    df["f0_copy"] = df["f0"] * 2 + 0.001
    df.loc[df.sample(frac=0.2, random_state=0).index, "f1"] = np.nan

    graph = build_graph()

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/graph.mmd").write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")

    t = time.time()
    state = graph.invoke({"df": df, "target": "target"})

    print("Plan données :", state["data_plan"].model_dump(exclude={"reasoning"}))
    print("Plan ML      :", state["ml_plan"].models)
    print("\nDécisions de la boucle :")
    for i, d in enumerate(state["trace"], 1):
        print(f"  {i}. {d.action} ({d.target_model})  corrections={d.corrections}")

    print("\nRésultats :")
    for r in state["results"]:
        print(f"  {r.model_name:<26} cv={r.cv_mean:.3f} ± {r.cv_std:.3f}  test={r.test_score:.3f}")
    print(f"\nModèle final : {state['final'].model_name}")

    print("\nTop features (SHAP) :")
    for f in state["explanation"].top_features[:4]:
        print(f"  {f.feature:<6} {f.mean_abs_shap:.4f}  {f.direction}")

    print("\nJustifications :")
    for f in state["research"].findings:
        print(f"  [{f.topic}] ({f.passage_id}) {f.note}")
    print("\nCorrections recherche :", state["research"].corrections)
    print(f"\nTemps total : {time.time() - t:.0f}s")