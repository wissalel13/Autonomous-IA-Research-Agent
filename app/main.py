import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from agents.clustering_report import render_clustering_markdown
from agents.orchestrator import build_graph
from agents.reporting import ReportingAgent
from llm.client import LLMClient
from tools.rag import build_index, search


@st.cache_resource
def ensure_index():
    try:
        search("test", k=1)
    except Exception:
        build_index()


ensure_index()

STEPS = {
    "profiling": "Profilage des données",
    "data_agent": "Data Analysis Agent",
    "ml_agent": "ML Agent : entraînement des modèles",
    "decide": "Décision de la boucle",
    "act": "Action (optimisation ou ajout de modèles)",
    "explain": "Explicabilité SHAP",
    "research_agent": "Research Agent (RAG)",
    "clustering": "Segmentation (clustering)",
}

st.set_page_config(page_title="AI Research Agent", layout="wide")
st.title("AI Research Agent")
st.caption("Analyse de données tabulaires : prédiction supervisée ou segmentation, avec justifications et limites.")

with st.sidebar:
    model = st.selectbox("Modèle LLM (Ollama)", ["qwen2.5:3b", "qwen2.5:7b"])
    st.caption("Le 7b est plus lent et gourmand en RAM.")
    file = st.file_uploader("Fichier CSV", type="csv")

if file is None:
    st.info("Charge un fichier CSV pour commencer.")
    st.stop()

df = pd.read_csv(file)
st.subheader("Aperçu")
st.dataframe(df.head(), use_container_width=True)
st.caption(f"{len(df)} lignes, {df.shape[1]} colonnes")

mode_label = st.radio(
    "Que veux-tu faire ?",
    ["Prédire une colonne", "Segmenter les données"],
    horizontal=True,
)
mode = "supervised" if mode_label == "Prédire une colonne" else "clustering"

exclude = st.multiselect("Colonnes à ignorer (identifiants, etc.)", list(df.columns))
target = None
if mode == "supervised":
    target = st.selectbox("Colonne cible", df.columns, index=len(df.columns) - 1)
else:
    st.caption("Aucune cible nécessaire. Si les données n'ont pas de groupes nets, l'outil le dira.")

if st.button("Lancer l'analyse", type="primary"):
    if mode == "supervised" and target in exclude:
        st.error("La colonne cible ne peut pas être ignorée.")
        st.stop()
    if mode == "supervised" and df[target].isna().any():
        st.error("La colonne cible contient des valeurs manquantes : supprime ces lignes d'abord.")
        st.stop()

    data = df.drop(columns=exclude)
    if mode == "clustering" and data.shape[1] < 2:
        st.error("Il faut au moins 2 colonnes pour segmenter.")
        st.stop()

    llm = LLMClient(model=model)
    graph = build_graph(llm)
    init = {"mode": mode, "df": data}
    if target:
        init["target"] = target
    state = dict(init)

    with st.status("Analyse en cours...", expanded=True) as status:
        t = time.time()
        for update in graph.stream(init, stream_mode="updates"):
            for node, out in update.items():
                state.update(out)
                st.write(f"Terminé : {STEPS.get(node, node)}")
        if mode == "supervised":
            report = ReportingAgent(llm).run(state)
        else:
            report = render_clustering_markdown(state["cluster_report"])
        status.update(label=f"Terminé en {time.time() - t:.0f}s", state="complete")

    st.session_state["result"] = {"mode": mode, "state": state, "report": report}

if "result" in st.session_state:
    res = st.session_state["result"]
    report, s = res["report"], res["state"]
    st.divider()

    if res["mode"] == "supervised":
        head, sep, tail = report.partition("## Justifications documentées")
        head = re.sub(r"!\[SHAP\]\(.*?\)\n?", "", head)
        st.markdown(head)
        st.image(s["explanation"].plot_path)
        st.markdown(sep + tail)
        st.download_button("Télécharger le rapport (.md)", report, file_name="report.md")
    else:
        cr = s["cluster_report"]
        report = re.sub(r"!\[Segments\]\(.*?\)\n?", "", report)
        head, sep, tail = report.partition("## Détail par segment")
        st.markdown(head)
        if cr.plot_path:
            st.image(cr.plot_path)
        st.markdown(sep + tail)
        st.download_button("Télécharger le rapport (.md)", report, file_name="segmentation.md")