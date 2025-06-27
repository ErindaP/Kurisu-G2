import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances
import ot
from copy import deepcopy
from faqtorisation.faq_src.utils.utils_graph.models import EmbeddingStore
from sklearn.metrics.pairwise import cosine_similarity



def compute_embeddings(graphs: List[nx.Graph], model: Optional[SentenceTransformer] = None) -> List[np.ndarray]:
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')

    
    embeddings: List[np.ndarray] = []

    for G in graphs:
        """print(G)
        for elm in G.nodes():
            print(elm, G.nodes[elm])
        """
        texts = [G.nodes[n]["text"] for n in G.nodes()]
        E = model.encode(texts, normalize_embeddings=True)
        embeddings.append(E)
        print("---------------------------")

    return embeddings




def normalize_distance(D: np.ndarray, factor: float = 1.5) -> np.ndarray:
    D = np.array(D)
    finite_vals = D[np.isfinite(D) & (D > 0)]

    penalty = np.percentile(finite_vals, 95) * factor if len(finite_vals) > 0 else 1.0
    D[np.isinf(D)] = penalty
    np.fill_diagonal(D, 0)
    return D


def compute_sim_all_nodes(G: nx.Graph, question: str, embeddings: EmbeddingStore = None, model: Optional[SentenceTransformer] = None) -> nx.Graph:
    if model is None:
        model = SentenceTransformer('model/all-MiniLM-L6-v2')

    texts = [G.nodes[n]["text"] for n in G.nodes()]
    if embeddings is None:
        embeddings = model.encode(texts, normalize_embeddings=True)
    
    question_embedding = model.encode([question], normalize_embeddings=True)[0]
    sim = cosine_similarity(embeddings, question_embedding.reshape(1, -1))
    print("similarity", sim)
    similarities = [float(s) for s in sim.flatten()]

    for i, node in enumerate(G.nodes()):
        G.nodes[node]["similarity"] = similarities[i]

    return G



