from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.data_analysis import DataAnalysisAgent
from agents.ml_agent import MLAgent, eligible_models
from agents.optimizer import base_name, decide, select_final
from agents.research import ResearchAgent
from llm.client import LLMClient
from schemas.data import DataProfile
from schemas.plan import DataAnalysisPlan, MLPlan, NextAction
from schemas.research import ResearchReport
from schemas.results import ExplanationResult, ModelResult
from tools.explain import explain_model, load_pipeline
from tools.modeling import run_experiments
from tools.profiling import profile_dataset
from tools.tuning import _split, tune_model
from agents.clustering_agent import ClusteringAgent
from schemas.clustering import ClusteringReport


class AgentState(TypedDict, total=False):
    mode: str  # "supervised" (défaut) ou "clustering"
    df: pd.DataFrame
    target: str
    profile: DataProfile
    data_plan: DataAnalysisPlan
    ml_plan: MLPlan
    results: list[ModelResult]
    trace: list[NextAction]
    next_action: NextAction
    final: ModelResult
    explanation: ExplanationResult
    research: ResearchReport
    cluster_report: ClusteringReport


def build_graph(llm: LLMClient | None = None, max_steps: int = 2, n_trials: int = 15):
    llm = llm or LLMClient()

    def profiling(s: AgentState):
        return {"profile": profile_dataset(s["df"], s["target"])}

    def data_agent(s: AgentState):
        return {"data_plan": DataAnalysisAgent(llm).run(s["profile"])}

    def ml_agent(s: AgentState):
        ml_plan, res = MLAgent(llm).run(s["df"], s["target"], s["profile"], s["data_plan"])
        return {"ml_plan": ml_plan, "results": res.results, "trace": []}

    def decide_next(s: AgentState):
        pool, _ = eligible_models(s["profile"], s["data_plan"])
        d = decide(
            llm, s["results"], s["profile"].target.task, s["data_plan"].primary_metric, pool
        )
        return {"next_action": d, "trace": s["trace"] + [d]}

    def act(s: AgentState):
        d, task = s["next_action"], s["profile"].target.task
        results = list(s["results"])
        if d.action == "optimize":
            results.append(
                tune_model(s["df"], s["target"], task, s["data_plan"], d.target_model, n_trials)
            )
        elif d.action == "add_models":
            pool, _ = eligible_models(s["profile"], s["data_plan"])
            tried = {base_name(r.model_name) for r in results}
            untried = [m for m in pool if m not in tried]
            results.extend(
                run_experiments(s["df"], s["target"], task, s["data_plan"], untried).results
            )
        return {"results": results}

    def explain(s: AgentState):
        task = s["profile"].target.task
        final = select_final(s["results"], s["data_plan"].primary_metric)
        X_train, X_test, _, _, _ = _split(s["df"], s["target"], task, s["data_plan"], 42)
        pipe = load_pipeline(final.run_id, final.model_name)
        expl = explain_model(
            pipe, X_train, X_test, final.model_name, run_id=final.run_id, task=task
        )
        return {"final": final, "explanation": expl}
    def research_agent(s: AgentState):
        report = ResearchAgent(llm).run(
            s["profile"], s["data_plan"], s["final"].model_name, explained=True
        )
        return {"research": report}

    def clustering(s: AgentState):
        return {"cluster_report": ClusteringAgent(llm).run(s["df"])}

    def route(s: AgentState) -> str:
        return "clustering" if s.get("mode") == "clustering" else "supervised"

    def after_decide(s: AgentState) -> str:
        return "explain" if s["next_action"].action == "stop" else "act"

    def after_act(s: AgentState) -> str:
        return "explain" if len(s["trace"]) >= max_steps else "decide"

    g = StateGraph(AgentState)
    g.add_node("profiling", profiling)
    g.add_node("data_agent", data_agent)
    g.add_node("ml_agent", ml_agent)
    g.add_node("decide", decide_next)
    g.add_node("act", act)
    g.add_node("explain", explain)
    g.add_node("research_agent", research_agent)
    g.add_node("clustering", clustering)

    g.add_conditional_edges(START, route, {"supervised": "profiling", "clustering": "clustering"})
    g.add_edge("clustering", END)
    g.add_edge("profiling", "data_agent")
    g.add_edge("data_agent", "ml_agent")
    g.add_edge("ml_agent", "decide")
    g.add_conditional_edges("decide", after_decide, {"explain": "explain", "act": "act"})
    g.add_conditional_edges("act", after_act, {"explain": "explain", "decide": "decide"})
    g.add_edge("explain", "research_agent")
    g.add_edge("research_agent", END)
    return g.compile()