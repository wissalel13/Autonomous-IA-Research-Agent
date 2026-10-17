from pathlib import Path

import numpy as np
import pandas as pd

from agents.clustering_agent import ClusteringAgent
from agents.clustering_report import render_clustering_markdown
from agents.reporting import _numbers
from tests.test_cat_weight import customers_df

if __name__ == "__main__":
    assert _numbers("29.6") == _numbers("29.60")
    assert _numbers("62,83") == _numbers("62.83")
    assert _numbers("63") != _numbers("62.83")
    print("comparaison de nombres : OK\n")

    customers, _ = customers_df()
    noise = pd.DataFrame(
        np.random.default_rng(0).uniform(size=(1500, 5)), columns=[f"x{i}" for i in range(5)]
    )
    agent = ClusteringAgent()
    Path("outputs").mkdir(exist_ok=True)

    for name, df in [("clients", customers), ("bruit", noise)]:
        md = render_clustering_markdown(agent.run(df))
        path = Path("outputs") / f"clustering_report_{name}.md"
        path.write_text(md, encoding="utf-8")
        print(f"--- {path} ---")
        print(md if name == "clients" else "\n".join(md.splitlines()[:14]))