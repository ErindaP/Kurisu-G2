import streamlit as st
import networkx as nx
import numpy as np
import faiss
import pyvis.network as net
import tempfile
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import streamlit.components.v1 as components
import requests
import json
from huggingface_hub import configure_http_backend
import os
import copy
# module to check time execution 
import time
from utils_streamlit.utils_st import draw_graph, draw_graph_with_similarity
from faq_src.utils.utils_graph.kg_base import compute_sim_all_nodes
from faq_src.utils.utils_graph.distance import compute_all_graphs_mds, compute_all_graphs_pca
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from faq_src.utils.utils_fgw.utils_fgw import compute_similarity_weighted_question_structure_distances
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

model_path = os.getenv("MODEL_PATH")
if not model_path:
    model_path = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------- SSL Bypass -----------------------------
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

# --------------------------------------------------------------

# --------------------- Streamlit App --------------------------
st.set_page_config(page_title="FAQtorisation", layout="wide")
st.title("FAQtorisation")

# 1. Graphe et embeddings 
uploaded_file = st.file_uploader(" Importer un graphe JSON", type=["json"])


# Utilisation du cache session pour stocker le graphe et le modèle (plus les embeddings)
if 'G' not in st.session_state:
    st.session_state['G'] = None
    st.session_state['model'] = None
    st.session_state['last_uploaded_filename'] = None

if uploaded_file is not None:
    # Vérifier si le fichier a changé
    if (st.session_state['last_uploaded_filename'] != uploaded_file.name):
        try:
            json_data = json.load(uploaded_file)
            G = nx.DiGraph()

            # Ajouter les nœuds
            for node in json_data["nodes"]:
                G.add_node(node["index"], **node)

            # Ajouter les arêtes
            for source, target in json_data["edges"]:
                G.add_edge(source, target, prob=1.0)

            # Charger le modèle une seule fois
            if st.session_state['model'] is None:
                model = SentenceTransformer(model_path)
                st.session_state['model'] = model
            else:
                model = st.session_state['model']

            st.session_state['G'] = G
            st.session_state['last_uploaded_filename'] = uploaded_file.name

            st.success(f"Graphe importé avec {len(G.nodes)} nœuds et {len(G.edges)} arêtes.")
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")
            st.stop()
    else:
        G = st.session_state['G']
        model = st.session_state['model']
        st.success(f"Graphe déjà chargé avec {len(G.nodes)} nœuds et {len(G.edges)} arêtes.")
else:
    st.info("💡 Importez un fichier JSON pour commencer.")
    st.stop()

# Recalculer les embeddings à chaque itération
embeddings = {n: st.session_state['model'].encode(G.nodes[n]["text"], normalize_embeddings=True) for n in G.nodes}

with st.expander(" Voir le graphe"):
    html_path = draw_graph(G)
    components.html(open(html_path, 'r', encoding='utf-8').read(), height=500, scrolling=True)


# Définir une liste de graphes avec similarité (On va ajouter petit à petit les graphes après avoir posé des questions)
if 'G_sim_list' not in st.session_state:
    st.session_state['G_sim_list'] = []

if 'last_question' not in st.session_state:
    st.session_state['last_question'] = None




question = st.text_input("Posez votre question")

if st.button("Ajouter un graphe avec similarité"):
    if not question:
        st.warning("Veuillez poser une question avant de calculer la similarité.")
        st.stop()

    # Vérifier si le graphe a déjà été calculé pour cette question
    if st.session_state['last_question'] == question:
        st.warning("Vous avez déjà posé cette question. Veuillez en poser une nouvelle.")
        st.stop()

    # Calculer la similarité pour tous les nœuds du graphe


    else:
        start_time = time.time()
        G_sim = compute_sim_all_nodes(G, question, model)
        G_sim.graph['question'] = question
        st.session_state['last_question'] = question
        

        st.session_state['G_sim_list'].append(copy.deepcopy(G_sim))
        end_time = time.time()
        st.success(f"Similarité calculée en {end_time - start_time:.2f} secondes.")
        with st.expander(" Voir le graphe avec similarité"):
            html_path = draw_graph_with_similarity(G_sim)
            components.html(open(html_path, 'r', encoding='utf-8').read(), height=500, scrolling=True)
        st.write("Nœuds avec similarité :")
        for n in G_sim.nodes(data=True):
            st.write(f"{n[0]}: {n[1].get('similarity', 0):.4f} - {n[1]['text']}")

disp_all = st.button("Afficher tous les graphes avec similarité")
if disp_all:
    if not st.session_state['G_sim_list']:
        st.warning("Aucun graphe avec similarité n'a été calculé.")
    else:
        for i, G_sim in enumerate(st.session_state['G_sim_list']):
            sim = [f"{G_sim.nodes[n]['similarity']:.4f}" for n in G_sim.nodes()]
            st.write(f"Graphe {i+1} - Nœuds avec similarité :")
            for n in G_sim.nodes(data=True):
                st.write(f"{n[0]}: {n[1].get('similarity', 0):.4f} - {n[1]['text']}")




# --------------------- FAQtorisation --------------------------


# Définition d'un dictionnaire de configurations par méthode:

method_configs = {
    "MDS": {
        "function": lambda graphs, n_components: compute_all_graphs_mds(graphs, n_components)
    },
    "PCA": {
        "function": lambda graphs, n_components: compute_all_graphs_pca(graphs, n_components)
    }

}



n_components = st.selectbox("Choisissez le nombre de dimensions pour la visualisation MDS", [2, 3], index=0)
method_proj = st.selectbox("Choisissez la méthode de projection", ["MDS", "PCA"], index=0)



# Partie projection et visualisation des graphes avec similarité 

if st.button("Visualiser les graphes avec similarité"):
    if not st.session_state['G_sim_list']:
        st.warning("Aucun graphe avec similarité n'a été calculé.")
    else:
        try:
            proj_results = method_configs[method_proj]["function"](st.session_state['G_sim_list'], n_components)
            print("G_sim_list:", st.session_state['G_sim_list'])
            print("proj_results:", proj_results)
            
            st.success("MDS calculé avec succès.")
            # Labels = question associée à chaque graphe
            labels = [graph.graph.get('question', f"Graphe {i+1}") for i, graph in enumerate(st.session_state['G_sim_list'])]
            print("Labels:", labels)
            if n_components == 2:
                fig = go.Figure(data=go.Scatter(
                    x=proj_results[:, 0],
                    y=proj_results[:, 1],
                    mode='markers+text',
                    text=labels,
                    textposition='top center',
                    marker=dict(size=10, color='royalblue')
                ))
                fig.update_layout(
                    title="Visualisation MDS 2D des graphes (similarité)",
                    xaxis_title="Dimension 1",
                    yaxis_title="Dimension 2"
                )
            else:  # 3D
                fig = go.Figure(data=go.Scatter3d(
                    x=proj_results[:, 0],
                    y=proj_results[:, 1],
                    z=proj_results[:, 2],
                    mode='markers+text',
                    text=labels,
                    textposition='top center',
                    marker=dict(size=5, color='royalblue')
                ))
                fig.update_layout(
                    title=f"Visualisation {method_proj} 3D des graphes (similarité)",
                    scene=dict(
                        xaxis_title="Dim 1",
                        yaxis_title="Dim 2",
                        zaxis_title="Dim 3"
                    )
                )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors de la visualisation interactive : {e}")