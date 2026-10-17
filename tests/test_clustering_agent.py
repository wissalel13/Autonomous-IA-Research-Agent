import sys
import time

import numpy as np
import pandas as pd

from agents.clustering_agent import ClusteringAgent
from llm.client import LLMClient
from tests.test_cat_weight import customers_df

if __name__ == "__main__":
    model = next((a for a in sys.argv[1:] if not a.startswith("--")), "qwen2.5:3b")
    print(f"Modèle : {model}")
    customers, _ = customers_df()
    noise = pd.DataFrame(
        np.random.default_rng(0).uniform(size=(1500, 5)), columns=[f"x{i}" for i in range(5)]
    )
    agent = ClusteringAgent(LLMClient(model=model), use_llm_titles="--llm" in sys.argv)

    for title, df in [("Clients (3 segments connus)", customers), ("Bruit (aucune structure)", noise)]:
        t = time.time()
        rep = agent.run(df)
        print(f"\n=== {title} ({time.time() - t:.0f}s) ===")
        print(rep.verdict)
        for s in rep.segments:
            print(f"  [{s.text_source}] {s.title} ({s.profile.share:.0%}) : {s.description}")
        for c in rep.corrections:
            print("  correction :", c)
        if "--llm" in sys.argv and rep.segments:
            n_llm = sum(s.text_source == "llm" for s in rep.segments)
            print(f"titres acceptés du LLM : {n_llm}/{len(rep.segments)}")
        