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
# module to check time execution 
import time
from kt_gen.knowledge_graph.utils.pydantic_models import Node, TraversalStep, EmbeddingStore, GraphWrapper, EmbeddingMatrix
from kt_gen.knowledge_graph.utils.utils_fgw import compute_embeddings, compute_structure_distances, compute_fgw, elementary_distance
from kt_gen.knowledge_graph.kg_fgw import recursive_traversal_with_scores_fusion_fgw_base
from kt_gen.knowledge_graph.kg_fgw import recursive_traversal_with_scores_fusion_fgw_enhanced,recursive_traversal_with_scores_fusion_fgw_genetic, recursive_traversal_with_score_fusion_fgw_genetic_optimized
from kt_gen.utils.streamlit.utils_st import draw_graph, display_graph_pyvis, display_graph_pyvis_with_prob, display_graph_distance, display_graph_distance_with_prob
from kt_gen.utils.llm.utils_llm import build_prompt, ask_llm
from kt_gen.knowledge_graph.kg_base_algo import recursive_traversal_with_scores, recursive_traversal_with_scores_fusion, recursive_traversal_with_scores_fusion_and_llm 
from kt_gen.knowledge_graph.utils.utils_fgw import compute_similarity_weighted_structure_distances, compute_embeddings
from huggingface_hub import hf_hub_download, snapshot_download
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

model_path = os.getenv("MODEL_PATH")
if not model_path:
    model_path = "utils/model/all-MiniLM-L6-v2"  # HF repo name



# --------------------- SSL Bypass ----------------------------- car wsl empechais de pull
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)


# Empêche transformers de tenter un téléchargement
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# --------------------------------------------------------------

# Initialisation
st.set_page_config(page_title="Graph LLM Navigator", layout="wide")
st.title(" Exploration d'un Graphe de Connaissance via LLM")

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
            if "structure_distances" in json_data:
                print("Structure distances found in JSON data.")
                for source, target, dict_data in json_data["edges"]:
                    G.add_edge(source, target, **dict_data)
                st.session_state['structure_distances'] = True
            else:
                for source, target in json_data["edges"]:
                    G.add_edge(source, target, prob=1.0)
            print("Yow la team")
            # Charger le modèle une seule fois
            if st.session_state['model'] is None:
                model = SentenceTransformer(model_path, trust_remote_code=False)

                st.session_state['model'] = model
            else:
                model = st.session_state['model']
            print("Yow la team 2")
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






##############################   Analyse des distances dans le graphe  ####################################
# Affichage de la distance structurelle : 
if st.button("Calculer la distance structurelle entre les nœuds"):
    embedding_struct = compute_embeddings([G], model)
    structure_distances = compute_similarity_weighted_structure_distances( 
        [G], 
        embedding_struct, 
        factor=1.5, 
        similarity_power=1.0
    )
    st.session_state['structure_distances'] = structure_distances

    st.success("Distances structurelles calculées.")
    # Affichage de la matrice de distance
    st.write("Matrice de distance structurelle :")
    for i, dist in enumerate(structure_distances):
        st.write(f"Graphe {i+1} :")
        st.dataframe(dist, use_container_width=True)

    # Comparer avec la distrance structurelle initiale: 
    base_structure_distances = compute_structure_distances([G], factor=1.5)
    st.write("Matrice de distance structurelle de base :")
    for i, dist in enumerate(base_structure_distances):
        st.write(f"Graphe {i+1} :")
        st.dataframe(dist, use_container_width=True)

    #print("Distances structurelles calculées :", structure_distances)
    # Ajouter les distances structurelles au graphe

    for i, node in enumerate(G.nodes):
        #print(f"Node: {node}, index: {i}")
        G.nodes[node]["structure_distance"] = structure_distances[0][i]

    for src, dst in G.edges:
        # Calculer la distance structurelle entre les nœuds source et destination
        src_index = list(G.nodes).index(src)
        dst_index = list(G.nodes).index(dst)
        structure_distance = structure_distances[0][src_index, dst_index]
        G.edges[src, dst]["structure_distance"] = structure_distance
# Affichage du graphe avec les distances structurelles
    display_graph_distance(G)



########################################################################################










# 4. LLM client
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# 5. Interface utilisateur
question = st.text_input("Pose ta question :", placeholder="Ex: Pourquoi les champignons ne font pas la photosynthèse ?")

# Définition d’un dictionnaire de configuration par méthode
method_configs = {
    "Méthode de fusion de base": {
        "function": lambda: recursive_traversal_with_scores_fusion("document", question, embedding_store, G, model, threshold=sim_threshold),
        "title": "méthode de fusion",
        "requires_threshold_slider": True,
        "disp_method" : lambda: display_graph_pyvis(G)

    },
    "Méthode de fusion with LLM": {
        "function": lambda: recursive_traversal_with_scores_fusion_and_llm("document", question, embedding_store, G, model, llm_awnser, threshold=sim_threshold, alpha=llm_ponderation, type_fuse="franchement"),
        "title": "méthode de fusion with LLM",
        "requires_llm_input": True,
        "requires llm_ponderation": True,
        "requires_threshold_slider": True,
        "disp_method" : lambda: display_graph_pyvis(G)

    },
    "Méthode de fusion sous distance FGW": {
        "function": lambda: recursive_traversal_with_scores_fusion_fgw_base("document", question, embedding_store, G, model, sim_threshold=sim_threshold, fgw_threshold=fgw_threshold, alpha=alpha_fgw),
        "title": "méthode de fusion sous distance FGW",
        "requires_threshold_slider": True,
        "requires_slider": True,
        "requires_alpha_fgw" : True,
        "disp_method" : lambda: display_graph_pyvis(G)

    },
    "Méthode de fusion avec FGW améliorée": {
        "function": lambda: recursive_traversal_with_scores_fusion_fgw_enhanced("document", question, embedding_store, G, model, sim_threshold=0.7, fgw_threshold=fgw_threshold, alpha=alpha_fgw,accelerated=False),
        "title": "méthode de fusion avec FGW améliorée",
        "requires_threshold_slider": True,
        "requires_slider": True,
        "requires_alpha_fgw" : True,
        "disp_method" : lambda: display_graph_pyvis(G)
    
    },
    "Méthode de fusion avec FGW améliorée et Evolution": {
        "function": lambda: recursive_traversal_with_scores_fusion_fgw_genetic("document", question, embedding_store, G, model, sim_threshold=0.7, fgw_threshold=fgw_threshold, alpha=alpha_fgw, structure_distance=structure_dist),
        "title": "méthode de fusion avec FGW améliorée et Evolution",
        "requires_threshold_slider": True,
        "requires_slider": True,
        "requires_alpha_fgw" : True,
        "requires_structure_distance": True,
        "disp_method" : lambda: display_graph_distance_with_prob(G),
    },
    "Méthode de fusion avec FGW améliorée et Evolution Optimisée": {
        "function": lambda: recursive_traversal_with_score_fusion_fgw_genetic_optimized("document", question, embedding_store, G, model, sim_threshold=0.7, fgw_threshold=fgw_threshold, alpha=alpha_fgw, structure_distance=structure_dist),
        "title": "méthode de fusion avec FGW améliorée et Evolution Optimisée",
        "requires_threshold_slider": True,
        "requires_slider": True,
        "requires_alpha_fgw" : True,
        "requires_structure_distance": True,
        "disp_method" : lambda: display_graph_distance_with_prob(G),
    },


}

# Sélecteur de méthode
method = st.selectbox("Sélectionner la méthode de recherche :", list(method_configs.keys()))
config = method_configs[method]

# Bouton pour lancer elementary_distance
if st.button("Calculer la distance élémentaire entre les nœuds"):
    distance= elementary_distance(G, embeddings, model)
    st.success("Distance: " + str(distance))


# Entrée texte si besoin
llm_awnser = None
if config.get("requires_llm_input"):
    llm_awnser = st.text_input("Réponse du LLM :", placeholder="Ex: Parce qu'ils n'ont pas de chlorophylle.")
    if not llm_awnser:
        st.warning("Veuillez entrer une réponse du LLM pour utiliser cette méthode.")
        st.stop()




# Slider si besoin
if config.get("requires_slider"):
    input_type = st.radio("Choisissez le mode d'entrée pour le seuil de distance FGW :", ("Slider", "Champ texte"))
    if input_type == "Slider":
        fgw_threshold = st.slider("Sélectionner le seuil de distance FGW :", min_value=0.0, max_value=2.0, value=0.3, step=0.01)
    else:
        fgw_threshold = st.number_input("Entrer la valeur du seuil de distance FGW :", min_value=0.0, max_value=2.0, value=0.3, step=0.01)
if config.get("requires_threshold_slider"):
    sim_threshold = st.slider("Sélectionner le seuil de similarité :", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
if config.get("requires_llm_ponderation"):
    llm_ponderation = st.slider("Sélectionner la pondération du LLM :", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
if config.get("requires_alpha_fgw"):
    alpha_fgw = st.slider("Sélectionner la valeur alpha pour FGW :", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
if config.get("requires_structure_distance"):
    structure_dist = st.selectbox("Sélectionner la distance de structure :", ["base","similarity_weighted"])





# Si une question est posée, on lance le traitement
if question:
    with st.spinner("Navigation dans le graphe..."):
        st.markdown(f"### Chemin suivi avec la {config['title']} :")

        # Initialisation du store d’embeddings
        embedding_store = EmbeddingStore(vectors=embeddings)

        # Lancement de la méthode choisie
        start_time = time.time()
        path_with_scores = config["function"]()
        end_time = time.time()
        st.success(f"Temps d'exécution : {end_time - start_time:.2f} secondes")

        # Affichage du graphe
        config["disp_method"]()
        # Récupération du dernier noeud pour le contexte
        node_ids = [nid for nid, _ in path_with_scores]
        #print(node_ids)
        last_node = node_ids[-1]
        context = last_node[1] + ": " + G.nodes[last_node[1]]["text"]

        st.markdown(f"## Contexte de la réponse : {context}")
        for elm in node_ids:
                st.markdown(f"- **{elm[1]}** : {G.nodes[elm[1]]['text']}")
        

        # Construction du prompt et appel au LLM
        prompt = build_prompt(question, context)
        response = ask_llm(prompt, client)

        st.markdown(f"### Réponse du LLM avec la {config['title']} :")
        st.success(response)

        # Affichage du chemin parcouru
        for nid, score in path_with_scores:
            text = G.nodes[nid]["text"]
            if score is not None:
                st.markdown(f"- **{nid}** (sim: `{score:.3f}`) : {text}")
            else:
                st.markdown(f"- **{nid}** (racine) : {text}")


