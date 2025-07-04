import streamlit as st
import json
import networkx as nx
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import time
from kt_gen.knowledge_graph.utils.pydantic_models import EmbeddingStore
from kt_gen.knowledge_graph.kg_fgw import recursive_traversal_with_scores_fusion_fgw_genetic, recursive_traversal_with_score_fusion_fgw_genetic_optimized
from kt_gen.utils.llm.utils_llm import build_prompt, ask_llm, resume_context
from kt_gen.utils.streamlit.utils_st import draw_graph, display_graph_distance_with_prob
from kt_gen.knowledge_graph.utils.utils_fgw import compute_embeddings, compute_similarity_weighted_structure_distances, compute_structure_distances
import streamlit.components.v1 as components
from kt_gen.knowledge_graph.utils.scrud_graph import export_graph_to_json, import_graph_from_json
from dotenv import load_dotenv
import os
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

# Chargement des variables d'environnement
load_dotenv()
model_path = os.getenv("MODEL_PATH")
if not model_path:
    model_path = "sentence-transformers/all-MiniLM-L6-v2"  # HF repo name

# streamlit: title = "Pipeline Tests - Graphe + LLM"
# Page config
st.set_page_config(page_title="Auto LLM Graphe", layout="wide")
st.title("Navigation Automatisée avec Graphe + LLM")

# --- Chargement du graphe ---
uploaded_graph = st.file_uploader("Charger un graphe JSON", type="json")
if uploaded_graph:
    json_data = json.load(uploaded_graph)
    G = nx.DiGraph()
    for node in json_data["nodes"]:
        G.add_node(node["index"], **node)
    if "structure_distances" in json_data:
        for source, target, dic in json_data["edges"]:
            G.add_edge(source, target, **dic)

    else: 
        for source, target in json_data["edges"]:
            G.add_edge(source, target, prob=1.0)
    st.success(f"Graphe chargé avec {len(G.nodes)} nœuds et {len(G.edges)} arêtes.")
else:
    st.stop()

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

# --- Initialisation modèle + LLM ---
model = SentenceTransformer(model_path, trust_remote_code=False)


client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
embeddings = {n: model.encode(G.nodes[n]["text"], normalize_embeddings=True) for n in G.nodes}
embedding_store = EmbeddingStore(vectors=embeddings)

# --- Paramètres ---
fgw_threshold = st.slider(" Seuil FGW", 0.0, 2.0, 0.3, 0.01)
sim_threshold = st.slider(" Seuil Similarité", 0.0, 1.0, 0.7, 0.01)
alpha = st.slider("Alpha", 0.0, 1.0, 0.5, 0.01)
lr = st.slider("Taux d'apprentissage", 0.0, 3.0, 0.1, 0.01)

# --- Lancement du processus ---


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


    #print("Distances structurelles calculées :", structure_distances)
    # Ajouter les distances structurelles au graphe
    """
    for i, node in enumerate(G.nodes):
        print(f"Node: {node}, index: {i}")
        G.nodes[node]["structure_distance"] = structure_distances[0][i]
    """
    for src, dst in G.edges:
        # Calculer la distance structurelle entre les nœuds source et destination
        src_index = list(G.nodes).index(src)
        dst_index = list(G.nodes).index(dst)
        structure_distance = structure_distances[0][src_index, dst_index]
        G.edges[src, dst]["structure_distance"] = structure_distance














if st.button("Lancer les questions en série"):
    for idx, question in enumerate(questions):
        st.subheader(f"🔹 Question {idx + 1} : {question}")
        print(f"Question {idx + 1}: {question}")
        #print("Graphe avant la navigation :", G.nodes(data=True))
        # Calculer les distances structurelles si ce n'est pas déjà fait
        # Check if structure_distances are present in the data of an edge:
        not_structure_distances = True
        for elm in G.edges(data=True):
            for key in elm[2].keys():
                if key == "structure_distance":
                    #print(f"Structure distance found in edge {elm[0]} -> {elm[1]}: {elm[2][key]}")
                    not_structure_distances = False
                    break
        if 'structure_distances' not in st.session_state and not_structure_distances:
            
            embedding_struct = compute_embeddings([G], model)
            structure_distances = compute_similarity_weighted_structure_distances(
                [G],
                embedding_struct,
                factor=1.5,
                similarity_power=1.0
            )
            st.session_state['structure_distances'] = structure_distances
            """
            # Ajouter les distances structurelles au graphe
            for i, node in enumerate(G.nodes):
                G.nodes[node]["structure_distance"] = structure_distances[0][i]
            """
            for src, dst in G.edges:
                src_index = list(G.nodes).index(src)
                dst_index = list(G.nodes).index(dst)
                structure_distance = structure_distances[0][src_index, dst_index]
                G.edges[src, dst]["structure_distance"] = structure_distance


        with st.spinner("Navigation..."):
            start = time.time()
            path = recursive_traversal_with_score_fusion_fgw_genetic_optimized(
                "document",
                question,
                embedding_store,
                G,
                model,
                sim_threshold=sim_threshold,
                fgw_threshold=fgw_threshold,
                alpha=alpha,
                structure_distance="similarity_weighted",
                lr=lr,
            )
            end = time.time()
            st.caption(f"Temps d'exécution : {end - start:.2f}s")
            print ("pas Normal... ")
        
        # Afficher le graphe
        display_graph_distance_with_prob(G)
        if st.session_state.get('graph') == None:
            st.session_state['graph'] = G
        else:
            st.session_state['graph'] = G
        node_ids = [nid for nid,_ in path]
        # Extraire contexte et poser la question au LLM
        last_node_id = node_ids[-1] 
        context = last_node_id[1] + ": " + G.nodes[last_node_id[1]]["text"] if last_node_id else ""
        #print(f"Context: {context}")
        st.markdown("### Contexte extrait")
        st.write(context)
        #for elm in node_ids:
            #print(f"Node ID: {elm[1]}, Text: {G.nodes[elm[1]]['text']}")
        """
        prompt = build_prompt(question, context, G)

        response = ask_llm(prompt, client)


        # Résultats
        st.markdown("### 🧾 Réponse LLM")
        st.success(response)
        """
# Boutton pour exporter le graphe
if st.button("Exporter le graphe au format JSON"):
    graph_json = export_graph_to_json(st.session_state['graph'])
    st.json(graph_json)
    st.download_button("Télécharger le graphe", data=json.dumps(graph_json, ensure_ascii=False, indent=2), file_name="graph.json", mime="application/json")


# Boutton génération axes principaux IA : 
if st.button("Générer les axes principaux IA"):
    if 'graph' in st.session_state:
        G = st.session_state['graph']
        response_node = {}
        for node in G.nodes:
            context = G.nodes[node]["text"]
            response = resume_context(context, client)
            response_node[node] = response
        st.session_state['response_node'] = response_node

        # Export in json
        response_json = {node: resp for node, resp in response_node.items()}
        st.json(response_json)
        st.download_button("Télécharger les réponses", data=json.dumps(response_json, ensure_ascii=False, indent=2), file_name="responses.json", mime="application/json")
