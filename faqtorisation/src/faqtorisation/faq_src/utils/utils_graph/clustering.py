import numpy as np 
from sklearn.cluster import KMeans
from typing import List, Dict, Optional, Tuple
import networkx as nx
from sentence_transformers import SentenceTransformer
from faqtorisation.faq_src.utils.utils_graph.models import EmbeddingStore
from sklearn.metrics import pairwise_distances



def cluster_space(
        coords: np.ndarray,
        n_clusters: int = 5,
        model: Optional[SentenceTransformer] = None,
) -> Tuple[List[int], List[float]]:
    """
    Cluster the coordinates in the embedding space using KMeans.

    Args:
        coords (np.ndarray): Coordinates of the nodes in the embedding space.
        n_clusters (int): Number of clusters to form.
        model (Optional[SentenceTransformer]): SentenceTransformer model for embeddings.

    Returns:
        Tuple[List[int], List[float]]: Cluster labels and their centroids.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(coords)
    centroids = kmeans.cluster_centers_

    return labels.tolist(), centroids.tolist()

