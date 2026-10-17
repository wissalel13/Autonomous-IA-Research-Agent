import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("benchmark/results.csv")
LOWER = {"rmse", "mae"}
TIE = 1e-4


def main(ref: str = "fixe"):
    r = pd.read_csv(RESULTS)
    # « gain » : toujours plus grand = meilleur (le RMSE est inversé)
    r["gain"] = np.where(r["eval_metric"].isin(LOWER), -r["test_score"], r["test_score"])

    print("Scores moyens (métrique d'évaluation de chaque jeu) :")
    print(r.groupby(["dataset", "config"])["test_score"].agg(["mean", "std", "count"]).round(4))

    rows = []
    for ds, g in r.groupby("dataset"):
        wide = g.pivot_table(index="seed", columns="config", values="gain")
        if ref not in wide:
            continue
        for cfg in wide.columns:
            if cfg == ref:
                continue
            d = (wide[cfg] - wide[ref]).dropna()
            if d.empty:
                continue
            rows.append(
                dict(
                    jeu=ds, config=cfg, graines=len(d),
                    ecart_moyen=round(d.mean(), 4),
                    ecart_type=round(d.std(ddof=1), 4) if len(d) > 1 else np.nan,
                    victoires=int((d > TIE).sum()),
                    egalites=int((d.abs() <= TIE).sum()),
                    defaites=int((d < -TIE).sum()),
                )
            )
    out = pd.DataFrame(rows)
    print(f"\nÉcarts appariés par graine, contre « {ref} » (positif = meilleur que la référence) :")
    print(out.to_string(index=False))

    lines = [
        f"| Jeu | Config | Graines | Écart moyen | Écart-type | Vict./Ég./Déf. |",
        "|---|---|---|---|---|---|",
    ]
    for _, x in out.iterrows():
        lines.append(
            f"| {x.jeu} | {x.config} | {x.graines} | {x.ecart_moyen:+} | {x.ecart_type} | "
            f"{x.victoires}/{x.egalites}/{x.defaites} |"
        )
    Path("benchmark/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nTableau markdown écrit dans benchmark/summary.md")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fixe")