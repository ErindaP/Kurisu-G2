import os
import streamlit as st
import networkx as nx
import numpy as np
import tempfile
import pyvis.network as net
import streamlit.components.v1 as components






# Visualisation interactive avec Pyvis
def draw_graph(G):
    net_graph = net.Network(height="500px", width="100%", directed=True)
    for n in G.nodes(data=True):
        net_graph.add_node(n[0], label=n[0], title=n[1]["text"], color="lightblue")
    for u, v in G.edges:
        net_graph.add_edge(u, v)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net_graph.save_graph(tmp_file.name)
    return tmp_file.name

def display_graph_pyvis(G):
    network = net.Network(height="600px", width="100%", directed=True, notebook=False)
    network.toggle_physics(True)

    # Ajoute tous les nœuds avec style
    for node in G.nodes:
        label = G.nodes[node].get("text", "")[:60] + "..."
        if G.nodes[node].get("type") == "fused":
            network.add_node(node, label="FUSION\n" + label, color="orange")
        else:
            network.add_node(node, label=label, color="lightblue")

    # Ajoute les arêtes
    for src, dst in G.edges:
        network.add_edge(src, dst)

    # Affiche dans Streamlit
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        path = tmp_file.name
        network.show(path, notebook=False)
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=600)
        os.remove(path)



def prob_to_color(prob):
    """
    Transforme une probabilité [0, 1] en couleur hexadécimale RGBA (rouge -> vert, avec transparence).
    """
    red = int((1 - prob) * 255)
    green = int(prob * 200)
    blue = 0
    alpha = int(prob * 255)  # Contrôle l'opacité
    return f"rgba({red}, {green}, {blue}, {prob:.2f})"  

def display_graph_pyvis_with_prob(G):
    network = net.Network(height="600px", width="100%", directed=True, notebook=False)
    network.toggle_physics(True)

    # Ajout des nœuds
    for node in G.nodes:
        label = G.nodes[node].get("text", "")[:60] + "..."
        if G.nodes[node].get("type") == "fused":
            network.add_node(node, label="FUSION\n" + label, color="orange")
        else:
            network.add_node(node, label=label, color="lightblue")

    # Ajout des arêtes avec mise en forme selon la probabilité
    for src, dst in G.edges:
        prob = G.edges[src, dst].get("prob", 1.0)
        color = prob_to_color(prob)
        width = 1 + 4 * prob  # Épaissir légèrement selon la proba
        network.add_edge(src, dst, color=color, width=width, title=f"Prob: {prob:.2f}")

    # Génération de la visualisation
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        path = tmp_file.name
        network.show(path, notebook=False)
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=600)
        os.remove(path)



def display_graph_distance(G):
    """
    Affiche le graphe avec la distance interne entre les noeuds: pour que ce soit plus lisible plus la distance entre deux noeuds est grande plus l'arête est longue (length )

    """
    network = net.Network(height="600px", width="100%", directed=True, notebook=False)
    network.toggle_physics(True)
    # Ajout des nœuds
    for node in G.nodes:
        label = G.nodes[node].get("text", "")[:60] + "..."
        if G.nodes[node].get("type") == "fused":
            network.add_node(node, label="FUSION\n" + label, color="orange")
        else:
            network.add_node(node, label=label, color="lightblue")
    # Ajout des arêtes avec distance
    for src, dst in G.edges:
        distance = G.edges[src, dst].get("structure_distance", 1.0)
        network.add_edge(src, dst, length=distance * 100, title=f"Distance: {distance:.2f}")
    # Génération de la visualisation
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        path = tmp_file.name
        network.show(path, notebook=False)
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=600)
        os.remove(path)


def display_graph_distance_with_prob(G):
    """
    Affiche le graphe avec la distance interne entre les noeuds et la probabilité de chaque arête.
    Plus la distance est grande, plus l'arête est longue, et la couleur de l'arête dépend de la probabilité.
    """
    network = net.Network(height="600px", width="100%", directed=True, notebook=False)
    network.toggle_physics(True)

    # Ajout des nœuds
    for node in G.nodes:
        label = G.nodes[node].get("text", "")[:60] + "..."
        if G.nodes[node].get("type") == "fused":
            network.add_node(node, label="FUSION\n" + label, color="orange")
        else:
            network.add_node(node, label=label, color="lightblue")

    # Ajout des arêtes avec distance et probabilité
    for src, dst in G.edges:
        distance = G.edges[src, dst].get("structure_distance", 1.0)
        print(f"src: {src}, dst: {dst}, distance: {distance}")
        prob = G.edges[src, dst].get("prob", 1.0)
        color = prob_to_color(prob)
        width = 1 + 4 * prob  # Épaissir légèrement selon la proba
        network.add_edge(src, dst, length=distance * 100, color=color, width=width,
                         title=f"Distance: {distance:.2f}, Prob: {prob:.2f}")

    # Génération de la visualisation
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        path = tmp_file.name
        network.show(path, notebook=False)
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=600)
        os.remove(path)



### J'aurais une partie refactoring à faire pour éviter la duplication de code entre les fonctions 






