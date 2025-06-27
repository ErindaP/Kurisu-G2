import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances
import ot
from copy import deepcopy
from kt_gen.knowledge_graph.utils.pydantic_models import EmbeddingStore
from sklearn.metrics.pairwise import cosine_similarity




def convert_ndarrays_to_lists(obj):
    if isinstance(obj, dict):
        return {k: convert_ndarrays_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays_to_lists(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    

def export_graph_to_json(graph: nx.Graph) -> Dict:
    """
    Export a NetworkX graph to a JSON-compatible dictionary.
    """
    def safe_node_data(n):
        data = dict(graph.nodes[n])
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = v.tolist()
        return {"index": n, **data}

    def safe_edge_data(u, v):
        data = dict(graph[u][v])
        for k, v_ in data.items():
            if isinstance(v_, np.ndarray):
                data[k] = v_.tolist()
        return (u, v, data)

    data = {
        "nodes": [safe_node_data(n) for n in graph.nodes()],
        "edges": [safe_edge_data(u, v) for u, v in graph.edges()],
        "structure_distances": {
            n: graph.nodes[n].get("structure_distance", 0.0).tolist()
            if isinstance(graph.nodes[n].get("structure_distance"), np.ndarray)
            else graph.nodes[n].get("structure_distance", 0.0)
            for n in graph.nodes()
        }
    }
    return data




def import_graph_from_json(data: Dict) -> nx.Graph:
    """
    Import a NetworkX graph from a JSON-compatible dictionary.
    """
    G = nx.DiGraph()
    for node in data["nodes"]:
        G.add_node(node["index"], **{k: v for k, v in node.items() if k != "index"})
    for u, v, attrs in data["edges"]:
        G.add_edge(u, v, **attrs)
    

    
    return G