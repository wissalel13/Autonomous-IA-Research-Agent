import time

from agents.orchestrator import build_graph
from tests.test_cat_weight import customers_df

if __name__ == "__main__":
    df, _ = customers_df()
    t = time.time()
    state = build_graph().invoke({"mode": "clustering", "df": df})

    rep = state["cluster_report"]
    print(rep.verdict)
    for s in rep.segments:
        print(f"  [{s.text_source}] {s.title} ({s.profile.share:.0%})")
    assert "profile" not in state and "results" not in state, "la branche supervisée a été exécutée"
    print(f"\nbranche supervisée non exécutée : OK ({time.time() - t:.0f}s)")