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
from kt_gen.knowledge_graph.kg_fgw import recursive_traversal_with_scores_fusion_fgw_enhanced,recursive_traversal_with_scores_fusion_fgw_genetic
from kt_gen.utils.streamlit.utils_st import draw_graph, display_graph_pyvis, display_graph_pyvis_with_prob, display_graph_distance, display_graph_distance_with_prob
from kt_gen.utils.llm.utils_llm import build_prompt, ask_llm, resume_context, summurize_axe
from kt_gen.knowledge_graph.kg_base_algo import recursive_traversal_with_scores, recursive_traversal_with_scores_fusion, recursive_traversal_with_scores_fusion_and_llm 
from kt_gen.knowledge_graph.utils.utils_fgw import compute_similarity_weighted_structure_distances, compute_embeddings

 # streamlit: title = "Pipeline IA"


def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Set the page configuration
st.set_page_config(
    page_title="Knowledge Graph - Interactive Analysis",
    page_icon=":bar_chart:",
    layout="wide"
)


# Un mode question et un mode lecture de fichiers avec questions et contextes en série : 

select_mode = st.selectbox(    "Sélectionnez le mode d'interaction",
    options=["Mode Question", "Mode Lecture de Fichiers"]
)

if select_mode == "Mode Lecture de Fichiers":
    st.write("Mode Lecture de Fichiers sélectionné. Veuillez télécharger un fichier contenant des questions et des contextes.")
    uploaded_file_questions = st.file_uploader("Téléchargez un fichier de questions (format JSON)", type=["json"])
    uploaded_file_contexts = st.file_uploader("Téléchargez un fichier de contextes (format JSON)", type=["json"])
    if uploaded_file_questions is not None and uploaded_file_contexts is not None:
        # Lire le contenu des fichiers JSON
        try:
            questions_data = json.load(uploaded_file_questions)
            questions = questions_data.get("questions", [])
            contexts_data = json.load(uploaded_file_contexts)
            contexts = contexts_data.get("contexts", [])
            st.write("Fichiers chargés avec succès.")
            for question in questions:
                st.write(f"Question : {question['question']}")
                prompt = build_prompt(question['question'], question.get('context', ''))
                reponse = ask_llm(prompt, client)
                st.success(f"Réponse du modèle : {reponse}")
                st.write("---")


        except json.JSONDecodeError as e:
                st.error(f"Erreur lors de la lecture des fichiers JSON : {e}")
    else:
        st.warning("Veuillez télécharger les fichiers de questions et de contextes pour continuer.")
    
    


else:
# Champ pour rentrer la question : 
    question = st.text_input("Posez votre question :")

    # Champ pour rentrer du contexte :
    context = st.text_area("Contexte (optionnel) :")


    # Lancer le model: 
    st.write("Envoi de la question au modèle...")


    if st.button("Envoyer la question et le contexte au modèle"):
        st.write("Question et contexte fournis. Envoi au modèle...")
        prompt = build_prompt(question, context)
        reponse= ask_llm(prompt, client)
        # Afficher la réponse
        st.write(f"Réponse du modèle à la question : {question}")
        st.success(reponse)
        st.write("---")
    if st.button("Résumer le contexte"):
        if context:
            st.write("Résumé du contexte...")
            summary = resume_context(context, client)
            st.write("Résumé :")
            st.success(summary)
        else:
            st.warning("Veuillez fournir un contexte pour le résumé.")
            
    
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

    if st.button("Générer les axes principaux LLM"):
        resp = summurize_axe(context, client)
        st.write("Axes principaux générés :")
        st.success(resp)
        # Export in json
        response_json = {"axes_principaux": resp}
        st.json(response_json)
        st.download_button("Télécharger les axes principaux", data=json.dumps(response_json, ensure_ascii=False, indent=2), file_name="axes_principaux.json", mime="application/json")
    
