import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances
import ot
from copy import deepcopy
from faq_src.utils.utils_graph.models import EmbeddingStore
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import ot
from faq_src.utils.utils_fgw.utils_fgw import compute_pairwise_gw, compute_pairwise_fgw,compute_embeddings

def L2_distance(
    x: np.ndarray,
    y: np.ndarray,
    metric: str = "euclidean",
) -> float:
    """
    Calculate the L2 distance between two vectors.

    Args:
        x (np.ndarray): First vector.
        y (np.ndarray): Second vector.
        metric (str): Metric to use for distance calculation. Default is 'euclidean'.

    Returns:
        float: The L2 distance between the two vectors.
    """
    return pairwise_distances(x.reshape(1, -1), y.reshape(1, -1), metric=metric)[0][0]



# MDS algorithm pour réduire la dimensionnalité des graphes
# On travaille ici dans des espaces avec n = nombre de noeuds dimensions 
# On veut représenter les graphes dans un espace de dimension 2 ou 3 pour la visualisation 

def vectorize_graph_by_similarities(G: nx.Graph, target_size: int = 20) -> np.ndarray:
    sims = np.array([G.nodes[n].get("similarity", 0) for n in G.nodes()])
    if len(sims) >= target_size:
        return sims[:target_size]
    else:
        return np.pad(sims, (0, target_size - len(sims)), constant_values=0)

def compute_all_graphs_mds(graph_list: list[nx.Graph], embeddings: EmbeddingStore, model: SentenceTransformer, n_components: int = 2, distance: str = "euclidean", alpha: float = 0.5) -> np.ndarray:
    vectors = [vectorize_graph_by_similarities(G) for G in graph_list]
    if distance == "gw":
        # Calcul des distances Gromov-Wasserstein
        dist_matrix = compute_pairwise_gw(graph_list, embeddings, model=model)
        print(f"dist_matrix : {dist_matrix}")
        print(f'Shape of dist_matrix: {dist_matrix.shape}')
    
    elif distance == "fgw":
        # Calcul des distances Fused Gromov-Wasserstein
        #store the values of the dict embeddings in a np.ndarray
        F1 = compute_embeddings([graph_list[0]], model=model)[0]
        EmbList = [F1 for _ in range(len(graph_list))]
        print("Yooooooooooooow la team")
        dist_matrix = compute_pairwise_fgw(graph_list, EmbList, model=model, alpha=alpha)
        print(f"dist_matrix : {dist_matrix}")
        #print(f'Shape of dist_matrix: {dist_matrix.shape}')

    else:
        # Calcul des distances euclidiennes
        if distance == "euclidean":
            dist_matrix = pairwise_distances(vectors, metric="cosine")
            print(f"dist_matrix : {dist_matrix}")
            print(f'Shape of dist_matrix: {dist_matrix.shape}')
        else:
            raise ValueError(f"Unsupported distance metric: {distance}")
    #dist_matrix = pairwise_distances(vectors, metric="euclidean")
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(dist_matrix)
    return coords



def compute_all_graphs_pca(graph_list: list[nx.Graph], embeddings: EmbeddingStore, model: SentenceTransformer, n_components: int = 2, distance: str = "euclidean", alpha: float = 0.5) -> np.ndarray:
    """
    PCA pour liste de graphs
    """

    vectors = [vectorize_graph_by_similarities(G) for G in graph_list]
    if distance == "gw":
        # Compute pairwise Gromov-Wasserstein distances
        dist_matrix = compute_pairwise_gw(graph_list, embeddings, model=model)
        print(f"dist_matrix : {dist_matrix}")
        print(f'Shape of dist_matrix: {dist_matrix.shape}')
    elif distance == "fgw":
        # Compute pairwise Fused Gromov-Wasserstein distances
        F1 = compute_embeddings([graph_list[0]], model=model)[0]
        EmbList = [F1 for _ in range(len(graph_list))]
        dist_matrix = compute_pairwise_fgw(graph_list, EmbList, model=model, alpha=alpha)
        print(f"dist_matrix : {dist_matrix}")
        #print(f'Shape of dist_matrix: {dist_matrix.shape}')
        
    else:
        # Compute pairwise Euclidean distances
        if distance == "euclidean":
            dist_matrix = pairwise_distances(vectors, metric="euclidean")
            print(f"dist_matrix : {dist_matrix}")
            print(f'Shape of dist_matrix: {dist_matrix.shape}')
        else:
            raise ValueError(f"Unsupported distance metric: {distance}")
    print(f"vectors : {vectors}")
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(dist_matrix)
    print(f"pca.explained_variance_ratio_: {pca.explained_variance_ratio_}")
    return coords


