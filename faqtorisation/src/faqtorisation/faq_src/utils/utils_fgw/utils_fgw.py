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



def compute_embeddings(graphs: List[nx.Graph], model: Optional[SentenceTransformer] = None) -> List[np.ndarray]:
    if model is None:
        model = SentenceTransformer('model/all-MiniLM-L6-v2')

    
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


def normalize_distance(D: np.ndarray, factor: float = 1.2, sim_power: float = 1.0  ) -> np.ndarray:
    D = np.array(D)

    penalty = np.exp(2 * sim_power) *factor
    D[np.isinf(D)] = penalty
    np.fill_diagonal(D, 0)
    
    return D


def normalize_distance_robust(D: np.ndarray, lower_percentile=5, upper_percentile=95) -> np.ndarray:
    D = np.array(D)
    finite_vals = D[np.isfinite(D)]

    v_min = np.percentile(finite_vals, lower_percentile)
    v_max = np.percentile(finite_vals, upper_percentile)

    # Clamp values for robustness
    D_clipped = np.clip(D, v_min, v_max)
    D_norm = (D_clipped - v_min) / (v_max - v_min + 1e-8)
    np.fill_diagonal(D_norm, 0)

    return D_norm



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
            # Plus la similarité est grande, plus la distance doit être petite
            adjusted_weight = 1.0 / (sim ** similarity_power + 1e-6)  # éviter div/0
            G_weighted.add_edge(u, v, weight=adjusted_weight)
            #print(f"Edge ({u}, {v}) - Similarity: {sim:.4f}, Adjusted Weight: {adjusted_weight:.4f}")    

        # Matrice des plus courts chemins (pondérée)
        D = nx.floyd_warshall_numpy(G_weighted, weight='weight')
        D = normalize_distance(D, factor=factor)
        weighted_dists.append(D)

    return weighted_dists

def compute_similarity_weighted_question_structure_distances(
    graphs: List[nx.Graph],
    embeddings: List[np.ndarray],
    question: str,
    model: Optional[SentenceTransformer] = None,
    factor: float = 1.2,
    similarity_power: float = 1.0
) -> List[np.ndarray]:
    """ Distance structure entre les noeuds dans le graphe, pondérée par la similarité des embeddings des noeuds avec une question.
        Plus la similarité d'un noeud avec la question est grande, plus la distance entre lui et son parent doit être petite.
    """


    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')

    question_embedding = model.encode([question], normalize_embeddings=True)[0]
    weighted_dists = []

    for G, E in zip(graphs, embeddings):
        # Création d'une copie pondérée du graphe
        G_weighted = nx.Graph() if not G.is_directed() else nx.DiGraph()
        G_weighted.add_nodes_from(G.nodes(data=True))
        
        for u, v in G.edges():
            #print(f"Processing edge ({u}, {v})")
            #print(f"Node {u} embedding: {E[list(G.nodes).index(u)]}")
            #print(f"Node {v} embedding: {E[list(G.nodes).index(v)]}")

            sim_u = cosine_similarity([E[list(G.nodes).index(u)]], [question_embedding])[0, 0]
            sim_v = cosine_similarity([E[list(G.nodes).index(v)]], [question_embedding])[0, 0]
            # Plus la similarité est grande, plus la distance doit être petite
            transfo_sim = np.exp(sim_v)/np.exp(1)
            adjusted_weight = 1.0 / ((transfo_sim) ** similarity_power + 1e-6)  # éviter div/0   / On prend en compte la similarité des deux noeuds avec la question (peut-être juste le fils c'est mieux ?)
            G_weighted.add_edge(u, v, weight=adjusted_weight)
            #print(f"Edge ({u}, {v}) - Similarity: {sim_u:.4f}, {sim_v:.4f}, Adjusted Weight: {adjusted_weight:.4f}")

        # Matrice des plus courts chemins (pondérée)
        D = nx.floyd_warshall_numpy(G_weighted, weight='weight')
        D = normalize_distance(D, factor=factor)
        #D = normalize_distance_robust(D, lower_percentile=5, upper_percentile=95)  # Normalisation robuste jsp si c'est utile ici je l'ajoute pour le test 
        weighted_dists.append(D)

    return weighted_dists


def compute_fgw(
    F1: np.ndarray,
    D1: np.ndarray,
    F2: np.ndarray,
    D2: np.ndarray,
    alpha: float = 0.5
) -> Tuple[float, np.ndarray]:
    p = np.ones(F1.shape[0]) / F1.shape[0]
    q = np.ones(F2.shape[0]) / F2.shape[0]
    M = pairwise_distances(F1, F2, metric="euclidean")
    dist, log = ot.gromov.fused_gromov_wasserstein2(M, D1, D2, p, q, alpha=alpha, log=True)
    return dist, log



def compute_gw(D1: np.ndarray, D2: np.ndarray) -> Tuple[float, np.ndarray]:
    p= np.ones(D1.shape[0]) / D1.shape[0]
    q= np.ones(D2.shape[0]) / D2.shape[0]
    dist, _= ot.gromov.gromov_wasserstein2(D1, D2, p, q, log=True)
    return dist,None



# Calculs de GW deux à deux sur une liste de graphes
def compute_pairwise_gw(graphs: List[nx.Graph], embeddings: EmbeddingStore, model: SentenceTransformer, alpha: float = 0.5) -> np.ndarray:
    """ Calcule la distance GW entre tous les graphes de la liste, en utilisant les embeddings et les distances structurelles
        -   embeddings : EmbeddingStore contenant les embeddings des noeuds de chaque graphe
        -   model : SentenceTransformer utilisé pour calculer les embeddings
        - On suppose que les graphes ont déjà été enrichis avec les attributs 'text' et 'structure_distance'
    """

    pairwise_gw = np.zeros((len(graphs), len(graphs)))

    for i in range(len(graphs)):
        D1=graphs[i].graph.get('structure_distance', None)
        
        for j in range(i + 1, len(graphs)):
            D2 = graphs[j].graph.get('structure_distance', None)
            gw_dist, _ = compute_gw(D1, D2)
            pairwise_gw[i, j] = gw_dist
            pairwise_gw[j, i] = gw_dist  # Symétrique

# Ajoute une normalisation des distances GW
    #pairwise_gw = normalize_distance(pairwise_gw)
    """ ici c'est des tests pour voir si la normalisation est utile ou pas"""
    #pairwise_gw = np.log1p(pairwise_gw + 1e-6)  # Pour éviter les problèmes de log(0)
    #pairwise_gw = pairwise_gw / np.max(pairwise_gw)  # Normalisation entre 0 et 1

    

    return pairwise_gw


# Calculs de FGW deux à deux sur une liste de graphes
def compute_pairwise_fgw(graphs: List[nx.Graph], embeddings: List[EmbeddingStore], model: SentenceTransformer, alpha: float = 0.5) -> np.ndarray:
    """ Calcule la distance FGW entre tous les graphes de la liste, en utilisant les embeddings et les distances structurelles
        -   embeddings : EmbeddingStore contenant les embeddings des noeuds de chaque graphe
        -   model : SentenceTransformer utilisé pour calculer les embeddings
        - On suppose que les graphes ont déjà été enrichis avec les attributs 'text' et 'structure_distance'
    """

    pairwise_fgw = np.zeros((len(graphs), len(graphs)))

    for i in range(len(graphs)):
        F1 = embeddings[i]
        D1 = graphs[i].graph.get('structure_distance', None)
        #print(f"Embeddings for graph {i}: {F1}")
        
        for j in range(i + 1, len(graphs)):
            F2 = embeddings[j]
            D2 = graphs[j].graph.get('structure_distance', None)
            print("debug1")
            fgw_dist, _ = compute_fgw(F1, D1, F2, D2, alpha=alpha)
            print("debug2")
            pairwise_fgw[i, j] = fgw_dist
            pairwise_fgw[j, i] = fgw_dist  # Symétrique
    # Ajoute une normalisation des distances FGW
    #pairwise_fgw = normalize_distance(pairwise_fgw)
    """ ici c'est des tests pour voir si la normalisation est utile ou pas"""
    #pairwise_fgw = np.log1p(pairwise_fgw + 1e-6)  # Pour éviter les problèmes de log(0)
    #pairwise_fgw = pairwise_fgw / np.max(pairwise_fgw)  # Normalisation entre 0 et 1
    return pairwise_fgw



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
