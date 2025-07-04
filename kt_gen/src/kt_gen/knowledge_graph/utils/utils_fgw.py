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
from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv())
model_path = os.getenv("MODEL_PATH")
if not model_path:
    model_path = "sentence-transformers/all-MiniLM-L6-v2"  # HF repo name



def compute_embeddings(graphs: List[nx.Graph], model: Optional[SentenceTransformer] = None) -> List[np.ndarray]:
    if model is None:
        model = SentenceTransformer(model_path, trust_remote_code=False)

    
    embeddings: List[np.ndarray] = []

    for G in graphs:
        """print(G)
        for elm in G.nodes():
            print(elm, G.nodes[elm])
        """
        texts = [G.nodes[n]["text"] for n in G.nodes()]
        E = model.encode(texts, normalize_embeddings=True)
        embeddings.append(E)
        #print(f"Shape of embeddings for graph with {len(G.nodes())} nodes: {E.shape}")
        #print("---------------------------")

    return embeddings


def normalize_distance(D: np.ndarray, factor: float = 1.5) -> np.ndarray:
    D = np.array(D)
    finite_vals = D[np.isfinite(D) & (D > 0)]

    penalty = np.percentile(finite_vals, 95) * factor if len(finite_vals) > 0 else 1.0
    D[np.isinf(D)] = penalty
    np.fill_diagonal(D, 0)
    return D



def normalize_distance_exp(D: np.ndarray, factor: float = 1.5, sim_power: float = 1.0  ) -> np.ndarray:
    D = np.array(D)

    penalty = np.exp(2 * sim_power) *factor
    D[np.isinf(D)] = penalty
    D[D <= 0] = penalty  # Assure que les distances négatives sont traitées
    np.fill_diagonal(D, 0)
    
    return D


def compute_structure_distances(graphs: List[nx.Graph], factor: float = 1.5) -> List[np.ndarray]:
    dist: List[np.ndarray] = [np.array(nx.floyd_warshall_numpy(G)) for G in graphs]

    for i, G in enumerate(graphs):
        dist[i] = normalize_distance(dist[i], factor=factor)

    return dist


def compute_similarity_weighted_structure_distances(
    graphs: List[nx.Graph],
    embeddings: List[np.ndarray],
    factor: float = 1.5,
    similarity_power: float = 1.0
) -> List[np.ndarray]:
    
    """ Distance structure entre les noeuds dans le graphe, pondérée par la similarité des embeddings des noeuds."""

    weighted_dists = []

    for G, E in zip(graphs, embeddings):
        # Création d'une copie pondérée du graphe
        G_weighted = nx.Graph() if not G.is_directed() else nx.DiGraph()
        G_weighted.add_nodes_from(G.nodes(data=True))

        for u, v in G.edges():
            sim = cosine_similarity([E[list(G.nodes).index(u)]], [E[list(G.nodes).index(v)]])[0, 0]
            transfo_sim = np.exp(sim)/np.exp(1)  # Transformation exponentielle pour remettre entre [0, 1]
            # Plus la similarité est grande, plus la distance doit être petite
            adjusted_weight = 1.0 / ((transfo_sim) ** similarity_power + 1e-6)  # éviter div/0
            G_weighted.add_edge(u, v, weight=adjusted_weight)
            #print(f"Edge ({u}, {v}) - Similarity: {sim:.4f}, Adjusted Weight: {adjusted_weight:.4f}")    

        # Matrice des plus courts chemins (pondérée)
        #D = nx.floyd_warshall_numpy(G_weighted, weight='weight')
        D = nx.adjacency_matrix(G_weighted, weight='weight')
        D = D.toarray()
        D = normalize_distance_exp(D, factor=factor) # Probablement besoin d'une normalisation log + max ici ? 
        weighted_dists.append(D)

    return weighted_dists


def safe_cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 1/np.exp(2)  # ou une valeur par défaut
    return cosine_similarity([vec1], [vec2])[0, 0]




def compute_fgw(
    F1: np.ndarray,
    D1: np.ndarray,
    F2: np.ndarray,
    D2: np.ndarray,
    alpha: float = 0.5
) -> Tuple[float, np.ndarray]:
    p = np.ones(F1.shape[0]) / F1.shape[0]
    q = np.ones(F2.shape[0]) / F2.shape[0]
    D1 = D1.astype(np.float64)
    D2 = D2.astype(np.float64)
    factor=1.5
    if np.isnan(D1).any():
            print("NaN detected in distance matrix D1, replacing with penalty.")
            D1 = np.nan_to_num(D1, nan=np.exp(2) * factor)  # ou penalty défini dans normalize_distance_exp
    if np.isnan(D2).any():
            
            print("NaN detected in distance matrix D2, replacing with penalty.")
            #print(f"D2 before replacement: {D2}")
            D2 = np.nan_to_num(D2, nan=np.exp(2) * factor)  # ou penalty défini dans normalize_distance_exp

    p = p.astype(np.float64)
    q = q.astype(np.float64)
    M = pairwise_distances(F1, F2, metric="euclidean")
    M = M.astype(np.float64)  # seulement si utilisé


    def validate_input(D1, D2, p, q):
        assert D1.shape[0] == len(p), "p does not match D1 size"
        assert D2.shape[0] == len(q), "q does not match D2 size"
        assert np.all(p >= 0) and np.isclose(np.sum(p), 1), "p not in simplex"
        assert np.all(q >= 0) and np.isclose(np.sum(q), 1), "q not in simplex"


    validate_input(D1, D2, p, q)
    T, log = ot.gromov.fused_gromov_wasserstein(M, D1, D2, p, q, alpha=alpha, log=True)
    return log['fgw_dist'], T # REGARDER Si y a besoin de Normaliser d'une quelconque manière hum 



def compute_gw(D1: np.ndarray, D2: np.ndarray) -> Tuple[float, np.ndarray]:
    p= np.ones(D1.shape[0]) / D1.shape[0]
    q= np.ones(D2.shape[0]) / D2.shape[0]
    T, log = ot.gromov.gromov_wasserstein(D1, D2, p, q, log=True)
    return log['gw_dist'], T

# Algorithme de fusion de graphe sous couvert de distance FGW

def assign_depths(G: nx.DiGraph):
    """Assigne à chaque nœud de G un attribut 'depth', qui correspond à sa profondeur à partir d'une racine."""
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Le graphe doit être un DAG pour que la profondeur soit bien définie.")

    depths = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        if not preds:
            depths[node] = 0
        else:
            depths[node] = 1 + max(depths[pred] for pred in preds)
        G.nodes[node]["depth"] = depths[node]
## Fonction prenant un graphe et calculant la distance d'une opération élémentaire (fusion/suppresion d'un noeud) entre deux graphes

def elementary_distance(G1: nx.Graph, embeddings1: EmbeddingStore, model: SentenceTransformer, alpha: float = 0.5, operation: str = "fusion", iterations: int = 10) -> float:
    """ Calcule une copie du graphe G1, puis effectue une opération élémentaire et calcule la distance FGW entre le graphe modifié et le graphe d'origine G1: 
           -   Pour l'opération de fusion, sélectionne aléatoirement deux noeuds à fusionner (les 2 noeuds noeuds doivent être sur la même profondeur) et calcule la distance FGW entre le graphe modifié et le graphe d'origine G1.
               Pour plus de précision, on peut réaliser plusieurs itérations et renvoyer la moyenne des distances.
    """

    assign_depths(G1)

    if operation == "fusion":
        dist_tot = 0.0
        not_inf = 0.0
        for _ in range(iterations):
            # Sélection aléatoire de deux noeuds à fusionner
            nodes = list(G1.nodes())
            if len(nodes) < 2:
                return float('inf')
            max_depth = max([G1.nodes[n]["depth"] for n in G1.nodes()])
            p = np.random.randint(1, max_depth)
            candidates = [n for n in nodes if G1.nodes[n]["depth"] == p]
            if len(candidates) < 2:
                continue  # Pas assez de noeuds à fusionner
            node1, node2 = np.random.choice(candidates, size=2, replace=False)
            # Création d'une copie du graphe
            G2 = deepcopy(G1)
            # Fusion des deux noeuds
            fused_text = G2.nodes[node1]["text"] + " | " + G2.nodes[node2]["text"]
            new_node_id = f"fused_{node1}_{node2}"
            G2.add_node(new_node_id, text=fused_text, type="fused", depth=G2.nodes[node1]["depth"])
            # Reconnexion des petits-enfants
            for child in G2.successors(node1):
                G2.add_edge(new_node_id, child)
            for child in G2.successors(node2):
                G2.add_edge(new_node_id, child)
            # Suppression des noeuds fusionnés
            G2.remove_node(node1)
            G2.remove_node(node2)
            # Calcul de la distance FGW
            F1, F2 = compute_embeddings([G1, G2], model=model)
            D1, D2 = compute_structure_distances([G1, G2])

            fgw_dist, _ = compute_fgw(F1, D1, F2, D2, alpha=alpha)
            if not np.isinf(fgw_dist):
                dist_tot += fgw_dist
                not_inf += 1
            
        return dist_tot / not_inf if not_inf > 0 else float('inf')

            
        


def compute_node_features(G, embeddings):
    return np.array([embeddings[n] for n in G.nodes])

def compute_structure_matrix(G):
    return np.array(nx.floyd_warshall_numpy(G))
