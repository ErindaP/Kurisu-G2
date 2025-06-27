import streamlit as st
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import json
from kt_gen.knowledge_graph.extract_graph import structure_text



# Configuration initiale
st.set_page_config(page_title="Graphe de Connaissance (Inclusion Textuelle)", layout="centered")
st.title("Graphe de Connaissance par Inclusion Textuelle")

# Chargement de spaCy
@st.cache_resource
def load_spacy_model():
    return spacy.load("fr_core_news_sm")

nlp = load_spacy_model()



# Visualisation avec surbrillance d’un nœud sélectionné
def plot_graph(graph, selected_node=None, level="document"):
    pos = nx.spring_layout(graph, seed=42)
    node_colors = []
    node_sizes = []
    node_to_draw = []

    for node, data in graph.nodes(data=True):
        if data.get('type') == 'merged':
            node_to_draw.append([node, data])
        
        if level == "document" and data.get('type') == 'document':
            node_to_draw.append([node, data])
        elif level == "section" and data.get('type') in ['document', 'section']:
            node_to_draw.append([node, data])
        elif level == "paragraph" and data.get('type') in ['document', 'section', 'paragraph']:
            node_to_draw.append([node, data])
        elif level == "sentence" and data.get('type') in ['document', 'section', 'paragraph', 'sentence']:
            node_to_draw.append([node, data])
        else:
            continue


    newGraph = graph.subgraph([node for node, data in node_to_draw])
    pos = nx.spring_layout(newGraph, seed=42)

    for node, data in node_to_draw:                
        # Couleur selon le type
        


        if data.get('type') == 'document':
            color = 'lightblue'
        elif data.get('type') == 'section':
            color = 'lightgreen'
        elif data.get('type') == 'paragraph':
            color = 'lightyellow'
        elif data.get('type') == 'sentence':
            color = 'lightcoral'
        elif data.get('type') == 'merged':
            color = 'lightpink'
        else:
            color = 'gray'

        if node == selected_node:
            color = 'gold'
            size = 1000
        else:
            size = 800

        node_colors.append(color)
        node_sizes.append(size)
    

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(newGraph, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, font_size=8, ax=ax)
    plt.title("Graphe de Connaissance (nœud sélectionné en jaune)")
    return fig


# Fonction de fusion de noeuds (et sous-graphes) pour le graphe 
def merge_nodes(graph, node1, node2):
    # Créer un nouveau nœud fusionné
    merged_node = f"{node1}_{node2} "

    graph.add_node(merged_node, type="merged", text=f"Merged: {node1} and {node2}")

    # Ajouter des arêtes du nœud fusionné vers les voisins de node1 et node2
    for neighbor in list(graph.neighbors(node1)) + list(graph.neighbors(node2)):
        graph.add_edge(merged_node, neighbor)


    # Faire pointer l'arête qui allait sur node1 sur le nouveau nœud fusionné
    for pred in list(graph.predecessors(node1)):
        graph.add_edge(pred, merged_node)
        graph.remove_edge(pred, node1)


    # Supprimer les anciens nœuds
    graph.remove_node(node1)
    graph.remove_node(node2)

    return graph

# Interface utilisateur
with st.expander("Texte brut à analyser", expanded=True):
    sample = """1. Introduction

L'intelligence artificielle transforme de nombreux secteurs.
Elle repose sur l'apprentissage automatique.

2. Contexte

Les algorithmes évoluent rapidement.
Ils nécessitent de grandes quantités de données."""
    user_text = st.text_area("Texte brut :", value=sample, height=300)
if "graph" not in st.session_state:
    st.session_state.graph = None

if st.button("Générer le graphe"):
    if user_text.strip():
        with st.spinner("Analyse et génération du graphe..."):
            G = structure_text(user_text, nlp)
            st.session_state.graph = G
            st.session_state.node = list(G.nodes)[0]  # valeur par défaut

if st.session_state.graph:
    G = st.session_state.graph
    all_nodes = list(G.nodes)

    selected = st.selectbox("Sélectionner un nœud à explorer :", all_nodes, index=all_nodes.index(st.session_state.get("node", all_nodes[0])))
    st.session_state.node = selected

    # Ajout d'une selectbox pour choisir le niveau de ramification
    level = st.selectbox("Sélectionner le niveau de ramification :", ["document", "section", "paragraph", "sentence"], index=0)
    st.session_state.level = level

    fig = plot_graph(G, selected_node=selected, level=level)
    st.pyplot(fig)

    node_data = G.nodes[selected]
    st.markdown("### Détails du nœud sélectionné")
    st.markdown(f"Type : `{node_data.get('type', 'inconnu')}`")
    st.markdown("Contenu :")
    st.code(node_data.get("text", "(aucun texte)"), language="markdown")


## Fonction d'export de graphes sous la forme JSON avec un champ : index , un champ texte et une liste de arrêtes dirigées
def export_graph_to_json(graph,level="sentence"):

    nodes = []
    edges = []
    for node, data in graph.nodes(data=True):
        if level == "document" and data.get('type') == 'document':
            nodes.append({"index": node, "text": data.get("text", ""), "type": data.get("type", "")})
        elif level == "section" and data.get('type') in ['document', 'section']:
            nodes.append({"index": node, "text": data.get("text", ""), "type": data.get("type", "")})
        elif level == "paragraph" and data.get('type') in ['document', 'section', 'paragraph']:
            nodes.append({"index": node, "text": data.get("text", ""), "type": data.get("type", "")})
        elif level == "sentence" and data.get('type') in ['document', 'section', 'paragraph', 'sentence']:
            nodes.append({"index": node, "text": data.get("text", ""), "type": data.get("type", "")})
    # Edges comme coupl de (noeud1, noeud2)

    for source, target in graph.edges():
        # Vérifier si les deux noeuds sont dans la liste des noeuds nodes 
        if source in [n["index"] for n in nodes] and target in [n["index"] for n in nodes]:
            edges.append((source, target))
    return {"nodes": nodes, "edges": edges}


# Exporter le graphe au format JSON
if st.button("Exporter le graphe au format JSON"):
    if st.session_state.graph:
        graph_json = export_graph_to_json(st.session_state.graph, level=st.session_state.level)
        st.write("level:", st.session_state.level)
        st.json(graph_json)
    else:
        st.warning("Aucun graphe à exporter.")

if st.button("Télécharger le graphe au format JSON"):
    if st.session_state.graph:
        graph_json = export_graph_to_json(st.session_state.graph, level=st.session_state.level)
        st.download_button("Télécharger le graphe", data=json.dumps(graph_json, ensure_ascii= False, indent=2), file_name="graph.json", mime="application/json")
    else:
        st.warning("Aucun graphe à télécharger.")




### TODO importer dans l'interface de calcul de distance FGW 

### TODO : Check le cahier avec les idées pour en discuter. 

### Interface pour la fusion de nœuds :

if st.session_state.graph:
    st.subheader("Fusionner des nœuds")
    selected_node1 = st.selectbox("Sélectionner le premier nœud à fusionner :", all_nodes, index=0)
    selected_node2 = st.selectbox("Sélectionner le deuxième nœud à fusionner :", all_nodes, index=1)
    st.session_state.merged_graph = None

    if st.button("Fusionner les nœuds sélectionnés"):
        if selected_node1 != selected_node2:
            with st.spinner("Fusion des nœuds..."):
                # Définir un nouveau graphe : 
                

                G_fus = merge_nodes(G, selected_node1, selected_node2)
                st.session_state.merged_graph = G_fus

                st.success(f"Nœuds {selected_node1} et {selected_node2} fusionnés.")
        else:
            st.warning("Veuillez sélectionner deux nœuds différents pour la fusion.")

    # Mettre à jour la liste des nœuds après fusion
    if st.session_state.merged_graph is not None:
        all_nodes = list(st.session_state.merged_graph.nodes)
        selected = st.selectbox("Sélectionner un nœud à explorer après fusion :", all_nodes, index=all_nodes.index(st.session_state.get("node", all_nodes[0])))
        st.session_state.node = selected
        fig_fus = plot_graph(st.session_state.merged_graph, selected_node=selected, level=st.session_state.level)
        st.pyplot(fig_fus)
        node_data = st.session_state.merged_graph.nodes[selected]
        st.markdown("### Détails du nœud sélectionné après fusion")
        st.markdown(f"Type : `{node_data.get('type', 'inconnu')}`")
        st.markdown("Contenu :")
        st.code(node_data.get("text", "(aucun texte)"), language="markdown")
        st.markdown("### Détails du graphe après fusion")
        st.markdown(f"Nombre de nœuds : {len(st.session_state.merged_graph.nodes)}")
        st.markdown(f"Nombre d'arêtes : {len(st.session_state.merged_graph.edges)}")
        st.markdown("### Graphe après fusion")
        