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



# Visualistion avec Pyvis du graphe telle que la taille des noeuds soit proportionnelle à la similarité
def draw_graph_with_similarity(G):
    net_graph = net.Network(height="500px", width="100%", directed=True)
    for n in G.nodes(data=True):
        size = 10 + (n[1].get("similarity", 0) * 50)  # Ajuster la taille en fonction de la similarité
        net_graph.add_node(n[0], label=n[0], title=f'{n[1]["text"]}\nSimilarity: {n[1].get("similarity", "N/A")}', size=size, color="lightblue")
    for u, v in G.edges:
        net_graph.add_edge(u, v)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net_graph.save_graph(tmp_file.name)
    return tmp_file.name


def display_graph_distance(G):
    """
    Affiche le graphe avec la distance interne entre les noeuds: pour que ce soit plus lisible plus la distance entre deux noeuds est grande, plus l'arête est longue.
    """
    network = net.Network(height="500px", width="100%", directed=True)
    network.toggle_physics(True)
    for node in G.nodes:
        label = G.nodes[node].get("text", "")[:60] + "..."
        if G.nodes[node].get("type") == "fused":
            network.add_node(node, label="FUSION\n" + label, title=label, color="orange", size=20)
        else:
            network.add_node(node, label=label, title=label, color="lightblue", size=10)
    for src, dst in G.edges:
        distance = G.edges[src, dst].get("structure_distance", 1.0)
        network.add_edge(src, dst, length=distance * 100, title=f"Distance: {distance:.6f}")
    # Génération de la visualisation
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        path = tmp_file.name
        network.show(path, notebook=False)
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=600)
        os.remove(path)



