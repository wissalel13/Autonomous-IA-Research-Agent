import os

os.environ.setdefault("STATE_DIR", "benchmark/state")  # runs MLflow isolés des vrais runs
os.makedirs(os.environ["STATE_DIR"], exist_ok=True)

import argparse
import time
import traceback
from pathlib import Path

import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from agents.data_analysis import DataAnalysisAgent
from agents.ml_agent import MLAgent
from agents.optimizer import run_agent_loop, select_final
from benchmark.datasets import DATASETS
from llm.client import LLMClient
from schemas.plan import DataAnalysisPlan
from tools.explain import load_pipeline
from tools.modeling import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    get_scoring,
    run_experiments,
)
from tools.profiling import profile_dataset
from tools.tuning import tune_model

RESULTS = Path("benchmark/results.csv")
CONFIGS = ["fixe", "fixe_cols", "fixe_cw", "fixe_opt", "regles_sans_opt", "regles", "llm"]


class NoLLM:
    """Faux LLM : tous les agents retombent sur leurs règles déterministes."""

    model = "aucun"

    def structured(self, *args, **kwargs):
        raise ValueError("LLM désactivé")


def run_fixed(train, target, task, metric, variant=None):
    """Pipeline fixe. variant : None, 'cols' (colonnes supprimées par les règles) ou 'cw' (class_weight)."""
    plan = DataAnalysisPlan(
        columns_to_drop=[], numeric_imputation="median", imbalance_strategy="none",
        primary_metric=metric, reasoning="pipeline fixe",
    )
    if variant:
        rules = DataAnalysisAgent(NoLLM()).run(profile_dataset(train, target))
        if variant == "cols":
            plan = plan.model_copy(update={"columns_to_drop": rules.columns_to_drop})
        elif variant == "cw":
            plan = plan.model_copy(update={"imbalance_strategy": rules.imbalance_strategy})
    models = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
    res = run_experiments(train, target, task, plan, models)
    best = next(r for r in res.results if r.model_name == res.best_model)
    return plan, best, 0


def run_fixed_opt(train, target, task, metric, n_trials):
    """Pipeline fixe + même optimisation que l'agent, appliquée au meilleur modèle de base."""
    plan, best, _ = run_fixed(train, target, task, metric)
    tuned = tune_model(train, target, task, plan, best.model_name, n_trials)
    return plan, select_final([best, tuned], metric), 0


def run_agent(train, target, llm, n_trials, max_steps=2):
    profile = profile_dataset(train, target)
    data_plan = DataAnalysisAgent(llm).run(profile)
    ml_plan, res = MLAgent(llm).run(train, target, profile, data_plan)
    final, _, trace = run_agent_loop(
        train, target, profile, data_plan, res.results,
        llm=llm, max_steps=max_steps, n_trials=n_trials,
    )
    n_corr = (
        len(data_plan.corrections)
        + len(ml_plan.corrections)
        + sum(len(d.corrections) for d in trace)
    )
    return data_plan, final, n_corr


def evaluate(final, plan, test, target, metric, multiclass):
    """Score du modèle final sur le test externe, avec la métrique d'évaluation fixée."""
    pipe = load_pipeline(final.run_id, final.model_name)
    X_test = test.drop(columns=list(plan.columns_to_drop) + [target])
    scoring = get_scoring(metric, multiclass)
    sign = -1 if scoring.startswith("neg_") else 1
    return float(sign * get_scorer(scoring)(pipe, X_test, test[target]))


def append(row: dict):
    pd.DataFrame([row]).to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--configs", nargs="+", default=["fixe", "regles", "llm"], choices=CONFIGS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()

    names = {k: k for k in CONFIGS}
    names["llm"] = f"llm_{args.model}"
    done = set()
    if RESULTS.exists():
        old = pd.read_csv(RESULTS)
        done = set(zip(old.dataset, old.config, old.seed))

    for ds in args.datasets:
        spec = DATASETS[ds]
        df, task, metric = spec["load"](), spec["task"], spec["metric"]
        multiclass = task == "classification" and df["target"].nunique() > 2
        if task == "classification":
            df["target"] = LabelEncoder().fit_transform(df["target"])

        for seed in range(args.seeds):
            strat = df["target"] if task == "classification" else None
            train, test = train_test_split(df, test_size=0.3, stratify=strat, random_state=seed)
            train, test = train.reset_index(drop=True), test.reset_index(drop=True)

            for key in args.configs:
                cfg = names[key]
                if (ds, cfg, seed) in done:
                    continue
                t = time.time()
                try:
                    if key == "fixe":
                        plan, final, n_corr = run_fixed(train, "target", task, metric)
                    elif key in ("fixe_cols", "fixe_cw"):
                        plan, final, n_corr = run_fixed(
                            train, "target", task, metric, variant=key.split("_")[1]
                        )
                    elif key == "fixe_opt":
                        plan, final, n_corr = run_fixed_opt(train, "target", task, metric, args.trials)
                    elif key == "regles_sans_opt":
                        plan, final, n_corr = run_agent(train, "target", NoLLM(), args.trials, max_steps=0)
                    else:
                        llm = NoLLM() if key == "regles" else LLMClient(model=args.model)
                        plan, final, n_corr = run_agent(train, "target", llm, args.trials)
                    score = evaluate(final, plan, test, "target", metric, multiclass)
                except Exception:
                    print(f"ÉCHEC {ds} {cfg} seed={seed}")
                    traceback.print_exc()
                    continue

                sec = time.time() - t
                append(
                    dict(
                        dataset=ds, config=cfg, seed=seed, eval_metric=metric,
                        test_score=round(score, 5), final_model=final.model_name,
                        agent_metric=plan.primary_metric, imbalance=plan.imbalance_strategy,
                        n_dropped=len(plan.columns_to_drop), n_corrections=n_corr,
                        seconds=round(sec, 1),
                    )
                )
                print(f"{ds:<11} {cfg:<18} seed={seed}  {metric}={score:.4f}  {final.model_name}  ({sec:.0f}s)")


if __name__ == "__main__":
    main()