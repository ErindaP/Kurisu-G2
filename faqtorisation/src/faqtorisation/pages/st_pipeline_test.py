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
from faq_src.utils.utils_graph.clustering import cluster_space
from faq_src.utils.utils_fgw.utils_fgw import compute_similarity_weighted_question_structure_distances,compute_embeddings,compute_pairwise_gw,compute_pairwise_fgw
from faq_src.utils.utils_graph.distance import vectorize_graph_by_similarities
from utils_streamlit.utils_st import display_graph_distance

from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.colors




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

# --- Chargement des questions ---
uploaded_questions = st.file_uploader("❓ Charger un fichier de questions (JSON ou TXT)", type=["json", "txt"])
questions = []
if uploaded_questions:
    if uploaded_questions.name.endswith(".json"):
        questions = json.load(uploaded_questions)
    else:
        questions = uploaded_questions.read().decode("utf-8").splitlines()
    st.success(f"{len(questions)} questions chargées.")
else:
    st.stop()
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
                model = SentenceTransformer("/home/erinda/CGI/faqtorisation/faqtorisation/src/faqtorisation/model/all-MiniLM-L6-v2")
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
if 'embeddings' not in st.session_state:
    
    st.session_state['embeddings'] = embeddings

with st.expander(" Voir le graphe"):
    html_path = draw_graph(G)
    components.html(open(html_path, 'r', encoding='utf-8').read(), height=500, scrolling=True)




if 'last_question' not in st.session_state:
    st.session_state['last_question'] = None



# Définir une liste de graphes avec similarité (On va ajouter petit à petit les graphes après avoir posé des questions)
if 'G_sim_list' not in st.session_state:
    st.session_state['G_sim_list'] = []
    embedding_struct = compute_embeddings([G], model=model)
    #print(f"Embedding structure: {embedding_struct}")
    dist_graph = compute_similarity_weighted_question_structure_distances([G], embedding_struct, question="théorie des graphes", model=model)
    """print(f"Distance structurelle pour le graphe initial: {dist_graph[0]}")
    st.dataframe(dist_graph[0], use_container_width=True)
    G.add_node('test', text='Ceci est un test')
    dist_graph = compute_similarity_weighted_question_structure_distances([G], embedding_struct, question="théorie des graphes", model=model)
    print(f"Distance structurelle pour le graphe avec le nœud test: {dist_graph[0]}")
    st.dataframe(dist_graph[0], use_container_width=True)
    
    #first_test = model.encode([G.nodes['test']['text']])
    #embedding_struct2 = compute_embeddings([G], model=model)
    #print(f"Embedding structure 2: {embedding_struct2}")
    #print(f"First test embedding: {first_test}")
    #print(f"All nodes in G: {list(G.nodes())}")
    """
    for i, question in enumerate(questions):
        if not question:
            st.warning(f"Question {i+1} vide, elle sera ignorée.")
            continue

        # Vérifier si le graphe a déjà été calculé pour cette question
        if st.session_state['last_question'] == question:
            st.warning(f"Vous avez déjà posé la question {i+1}. Veuillez en poser une nouvelle.")
            continue
        else:

            # Calculer la similarité pour tous les nœuds du graphe
            print(f"Calcul de la similarité pour la question {i+1}: {question}")
            start_time = time.time()
            G_sim = compute_sim_all_nodes(G, question, embeddings=embedding_struct[0], model=model)
            G_sim.graph['question'] = question
            # Calculer les distances structurelles pondérées par la similarité
            dist_graph = compute_similarity_weighted_question_structure_distances([G_sim], embedding_struct, question, model)
            G_sim.graph['structure_distance'] = dist_graph[0]
            for j, node in enumerate(G_sim.nodes()):
                G_sim.nodes[node]["structure_distance"] = dist_graph[0][j]

            for src,dst in G_sim.edges():
                src_index = list(G_sim.nodes()).index(src)
                dst_index = list(G_sim.nodes()).index(dst)
                structure_dist= dist_graph[0][src_index, dst_index]
                G_sim.edges[src, dst]['structure_distance'] = structure_dist
            

            st.session_state['last_question'] = question


            st.session_state['G_sim_list'].append(copy.deepcopy(G_sim))
            end_time = time.time()
            st.success(f"Similarité pour la question {i+1} calculée en {end_time - start_time:.2f} secondes.")

    #print("G_sim_list TEST:", st.session_state['G_sim_list'])
# Affichage des distances structurelles dans les dataframes
for i, graph in enumerate(st.session_state['G_sim_list']):
                structure_distances = np.array([graph.nodes[n]["structure_distance"] for n in graph.nodes()])
                st.dataframe(structure_distances, use_container_width=True)

#for elm in st.session_state['G_sim_list']:
    #print(f'Structure distances for question {elm.graph["question"]}:')
    #for n in elm.nodes(data=True):
        #print(f"Node {n[0]}: {n}")
    

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



n_components = st.selectbox("Choisissez le nombre de dimensions pour la visualisation MDS", [2, 3], index=0)
method_proj = st.selectbox("Choisissez la méthode de projection", ["MDS", "PCA"], index=0)
dist_choice = st.selectbox("Choisissez la distance pour la projection", ["gw", "fgw", "euclidean"], index=0)
alpha = st.slider("Alpha pour la fusion des graphes", min_value=0.0, max_value=1.0, value=0.5, step=0.1)



method_configs = {
    "MDS": {
        "function": lambda graphs, n_components: compute_all_graphs_mds(graphs, embeddings=st.session_state['embeddings'], model=st.session_state['model'], n_components=n_components, distance=dist_choice, alpha=alpha)
    },
    "PCA": {
        "function": lambda graphs, n_components: compute_all_graphs_pca(graphs, embeddings=st.session_state['embeddings'], model=st.session_state['model'], n_components=n_components, distance=dist_choice, alpha=alpha)
    }

}


n_clusters = st.number_input("Nombre de clusters pour le KMeans", min_value=1, max_value=10, value=3, step=1)
n_questions_per_cluster = st.number_input("Nombre de questions par cluster", min_value=1, max_value=10, value=5, step=1)
# Partie projection et visualisation des graphes avec similarité

if st.button("Visualiser les graphes avec similarité"):
    if not st.session_state['G_sim_list']:
        st.warning("Aucun graphe avec similarité n'a été calculé.")
    else:
        try:
            print(f'alpha: {alpha}')
            dist_pairwise_gw = compute_pairwise_gw(st.session_state['G_sim_list'], embeddings=st.session_state['embeddings'], model=st.session_state['model'])
            F1 = compute_embeddings([st.session_state['G_sim_list'][0]], model=st.session_state['model'])[0]
            EmbList = [F1 for _ in range(len(st.session_state['G_sim_list']))]
            dist_pairwise_fgw = compute_pairwise_fgw(st.session_state['G_sim_list'], embeddings=EmbList, model=st.session_state['model'], alpha=alpha)

            #display the pairwise distances
            st.session_state['dist_pairwise'] = dist_pairwise_gw
            st.session_state['dist_pairwise_fgw'] = dist_pairwise_fgw
            st.write("Distances deux à deux fused Gromov-Wasserstein:")
            st.dataframe(dist_pairwise_fgw, use_container_width=True)
            st.write("Distances deux à deux Gromov Wasserstein:")
            st.dataframe(dist_pairwise_gw, use_container_width=True)
            vectors = [vectorize_graph_by_similarities(G) for G in st.session_state['G_sim_list']]
            dist_pairwise_euclidean = pairwise_distances(vectors, metric="euclidean")
            st.session_state['dist_pairwise_euclidean'] = dist_pairwise_euclidean
            st.write("Distances deux à deux Euclidiennes:")
            st.dataframe(dist_pairwise_euclidean, use_container_width=True)
            proj_results = method_configs[method_proj]["function"](st.session_state['G_sim_list'], n_components)
            st.success(f"{method_proj} calculé avec succès.")

            labels = [graph.graph.get('question', f"Graphe {i+1}") for i, graph in enumerate(st.session_state['G_sim_list'])]
            # Affichage des graphes calculés avec la fonction display_graph_distance
            for graph in st.session_state['G_sim_list']:
                display_graph_distance(graph)
            

            # Couleurs par blocs de 8 questions
            color_palette = plotly.colors.qualitative.Plotly
            n_colors = len(color_palette)
            colors = [color_palette[(i // n_questions_per_cluster) % n_colors] for i in range(len(labels))]

            # Clustering KMeans sur les coordonnées projetées
            cluster_labels, _ = cluster_space(proj_results, n_clusters=n_clusters)
            n_clusters = len(set(cluster_labels))

            # Formes différentes par cluster (pour la 2D uniquement)
            symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'triangle-left', 'triangle-right']
            cluster_shapes = [symbols[cl % len(symbols)] for cl in cluster_labels]

            if n_components == 2:
                fig = go.Figure(data=go.Scatter(
                    x=proj_results[:, 0],
                    y=proj_results[:, 1],
                    mode='markers+text',
                    text=labels,
                    textposition='top center',
                    marker=dict(
                        size=10,
                        color=colors,
                        symbol=cluster_shapes,
                        line=dict(width=1, color='black')
                    ),
                ))
                fig.update_layout(
                    title=f"Visualisation {method_proj} 2D des graphes (clusters + similarité)",
                    xaxis_title="Dimension 1",
                    yaxis_title="Dimension 2"
                )
            else:  # 3D : on trace un Scatter3d par cluster
                fig = go.Figure()
                for cl in sorted(set(cluster_labels)):
                    indices = [i for i, c in enumerate(cluster_labels) if c == cl]
                    fig.add_trace(go.Scatter3d(
                        x=proj_results[indices, 0],
                        y=proj_results[indices, 1],
                        z=proj_results[indices, 2],
                        mode='markers+text',
                        name=f"Cluster {cl}",
                        text=[labels[i] for i in indices],
                        textposition='top center',
                        marker=dict(
                            size=6,
                            color=[colors[i] for i in indices],
                            line=dict(width=1, color='black'),
                            symbol='circle'  # Une seule forme en 3D
                        )
                    ))
                fig.update_layout(
                    title=f"Visualisation {method_proj} 3D des graphes (clusters + similarité)",
                    scene=dict(
                        xaxis_title="Dim 1",
                        yaxis_title="Dim 2",
                        zaxis_title="Dim 3"
                    )
                )

            st.plotly_chart(fig, use_container_width=True)

            # Affichage des clusters
            for cl in sorted(set(cluster_labels)):
                st.markdown(f"**Cluster {cl}:**")
                for idx, label in enumerate(labels):
                    if cluster_labels[idx] == cl:
                        st.write(f"- {label}")

        except Exception as e:
            st.error(f"Erreur lors de la visualisation interactive : {e}")
