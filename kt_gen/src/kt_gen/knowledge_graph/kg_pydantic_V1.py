import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances
import ot
from copy import deepcopy


# --- Pydantic Models ---

class Node(BaseModel):
    index: str
    text: str
    type: str = Field(default="unknown")


class TraversalStep(BaseModel):
    node_id: str
    similarity: Optional[float]  # None pour la racine


class EmbeddingStore(BaseModel):
    vectors: Dict[str, np.ndarray]
    model_config = ConfigDict(arbitrary_types_allowed=True)



class GraphWrapper(BaseModel):
    graph: nx.Graph

    class Config:
        arbitrary_types_allowed = True


class EmbeddingMatrix(BaseModel):
    vectors: np.ndarray

    class Config:
        arbitrary_types_allowed = True



# --- Recursive traversal (greedy) ---

def recursive_traversal_with_scores(
    start_node: str,
    question: str,
    embeddings: EmbeddingStore,
    G: nx.DiGraph,
    model: SentenceTransformer,
    max_depth: int = 3
) -> Tuple[List[TraversalStep], List[float]]:

    path: List[TraversalStep] = [TraversalStep(node_id=start_node, similarity=None)]
    current = start_node
    query_vec = model.encode([question], normalize_embeddings=True)[0]
    all_scores: List[float] = []

    for _ in range(max_depth):
        children = list(G.successors(current))
        if not children:
            break
        child_vecs = [embeddings.vectors[c] for c in children]
        sims = np.dot(child_vecs, query_vec)
        print(f"Similarité pour {current} : {sims}")

        best_idx = int(np.argmax(sims))
        current = children[best_idx]
        sim_score = float(sims[best_idx])
        path.append(TraversalStep(node_id=current, similarity=sim_score))
        all_scores.append(sim_score)

    return path, all_scores


# --- Recursive traversal with fusion ---

def recursive_traversal_with_scores_fusion(
    start_node: str,
    question: str,
    embeddings: EmbeddingStore,
    G: nx.DiGraph,
    model: SentenceTransformer,
    max_depth: int = 3,
    threshold: float = 0.8
) -> List[TraversalStep]:

    path: List[TraversalStep] = [TraversalStep(node_id=start_node, similarity=None)]
    current = start_node
    query_vec = model.encode([question], normalize_embeddings=True)[0]

    for depth in range(max_depth):
        children = list(G.successors(current))
        if not children:
            break

        child_vecs = [embeddings.vectors[c] for c in children]
        sims = np.dot(child_vecs, query_vec)
        print(f"Similarité pour {current} : {[f'{c}: {s:.3f}' for c, s in zip(children, sims)]}")

        max_sim = np.max(sims)
        similar_nodes = [i for i, sim in enumerate(sims) if sim > threshold * max_sim]

        if len(similar_nodes) > 1:
            new_node_id = f"fused_{current}_{depth}"
            fused_texts = " | ".join([G.nodes[children[i]]["text"] for i in similar_nodes])
            G.add_node(new_node_id, text=fused_texts, type="fused")

            # Embedding du nœud fusionné
            embeddings.vectors[new_node_id] = model.encode([fused_texts], normalize_embeddings=True)[0]

            # Reconnecte les petits-enfants au nœud fusionné
            for i in similar_nodes:
                sim_child = children[i]
                grand_children = list(G.successors(sim_child))
                for gc in grand_children:
                    G.add_edge(new_node_id, gc)
                G.remove_edge(current, sim_child)

            # Connecter le nouveau nœud au parent
            G.add_edge(current, new_node_id)

            # Supprimer les nœuds fusionnés
            
            for i in similar_nodes:
                G.remove_node(children[i])

            current = new_node_id
            mean_score = float(np.mean([sims[i] for i in similar_nodes]))
            path.append(TraversalStep(node_id=current, similarity=mean_score))

        else:
            best_idx = int(np.argmax(sims))
            current = children[best_idx]
            sim_score = float(sims[best_idx])
            path.append(TraversalStep(node_id=current, similarity=sim_score))

    return path





# --- Recursive traversal with fusion and LLM pre awnser embedding ---


def recursive_traversal_with_scores_fusion_and_llm(
    start_node: str,
    question: str,
    embeddings: EmbeddingStore,
    G: nx.DiGraph,
    model: SentenceTransformer,
    llm_awnser: str,
    max_depth: int = 3,
    threshold: float = 0.8,
    alpha: float = 0.5,
    type_fuse: str = "mean"
) -> List[TraversalStep]:
   path: List[TraversalStep] = [TraversalStep(node_id=start_node, similarity=None)]
   current = start_node
   query_vec = model.encode([question], normalize_embeddings=True)[0]
   llm_vec = model.encode([llm_awnser], normalize_embeddings=True)[0]
   # Deux options :
   # Soit on fusionne les embeddings de la question et de la réponse LLM (somme vectorielle pondérée) des deux 
   # Soit on regarde une moyenne pondérée des similarités de la question et de la réponse LLM avec les noeuds enfants
   if type_fuse == "mean":
       for _ in range(max_depth):
        children = list(G.successors(current))
        if not children:
            break
        child_vecs = [embeddings.vectors[c] for c in children]
        sims_question = np.dot(child_vecs, query_vec)
        sims_llm = np.dot(child_vecs, llm_vec)
        sims = alpha * sims_question + (1 - alpha) * sims_llm
        print(f"Similarité pour {current} : {sims}")
        max_sim = np.max(sims)
        similar_nodes = [i for i, sim in enumerate(sims) if sim > threshold * max_sim]
        if len(similar_nodes) > 1:
            new_node_id = f"fused_{current}_{_}"
            fused_texts = " | ".join([G.nodes[children[i]]["text"] for i in similar_nodes])
            G.add_node(new_node_id, text=fused_texts, type="fused")

            # Embedding du nœud fusionné
            embeddings.vectors[new_node_id] = model.encode([fused_texts], normalize_embeddings=True)[0]

            # Reconnecte les petits-enfants au nœud fusionné
            for i in similar_nodes:
                sim_child = children[i]
                grand_children = list(G.successors(sim_child))
                for gc in grand_children:
                    G.add_edge(new_node_id, gc)
                G.remove_edge(current, sim_child)

            # Connecter le nouveau nœud au parent
            G.add_edge(current, new_node_id)

            # Supprimer les nœuds fusionnés
            
            for i in similar_nodes:
                G.remove_node(children[i])

            current = new_node_id
            mean_score = float(np.mean([sims[i] for i in similar_nodes]))
            path.append(TraversalStep(node_id=current, similarity=mean_score))

        else:
            best_idx = int(np.argmax(sims))
            current = children[best_idx]
            sim_score = float(sims[best_idx])
            path.append(TraversalStep(node_id=current, similarity=sim_score))
   else:
       tot_vec = alpha * query_vec + (1 - alpha) * llm_vec
       for _ in range(max_depth):
          children = list(G.successors(current))
          if not children:
                break
          child_vecs = [embeddings.vectors[c] for c in children]
          sims = np.dot(child_vecs, tot_vec)
          print(f"Similarité pour {current} : {sims}")
          max_sim = np.max(sims)
          similar_nodes = [i for i, sim in enumerate(sims) if sim > threshold * max_sim]
          if len(similar_nodes) > 1:
                new_node_id = f"fused_{current}_{_}"
                fused_texts = " | ".join([G.nodes[children[i]]["text"] for i in similar_nodes])
                G.add_node(new_node_id, text=fused_texts, type="fused")
    
                # Embedding du nœud fusionné
                embeddings.vectors[new_node_id] = model.encode([fused_texts], normalize_embeddings=True)[0]
    
                # Reconnecte les petits-enfants au nœud fusionné
                for i in similar_nodes:
                 sim_child = children[i]
                 grand_children = list(G.successors(sim_child))
                 for gc in grand_children:
                      G.add_edge(new_node_id, gc)
                 G.remove_edge(current, sim_child)
    
                # Connecter le nouveau nœud au parent
                G.add_edge(current, new_node_id)
    
                # Supprimer les nœuds fusionnés
                
                for i in similar_nodes:
                 G.remove_node(children[i])
    
                current = new_node_id
                mean_score = float(np.mean([sims[i] for i in similar_nodes]))
                path.append(TraversalStep(node_id=current, similarity=mean_score))
    
          else:
                best_idx = int(np.argmax(sims))
                current = children[best_idx]
                sim_score = float(sims[best_idx])
                path.append(TraversalStep(node_id=current, similarity=sim_score))
    # On renvoie le chemin


   return path




# ---- Partie calcul de distance --- 


def compute_embeddings(graphs: List[nx.Graph], model: Optional[SentenceTransformer] = None) -> List[np.ndarray]:
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')

    
    embeddings: List[np.ndarray] = []

    for G in graphs:
        texts = [G.nodes[n]["text"] for n in G.nodes()]
        E = model.encode(texts, normalize_embeddings=True)
        embeddings.append(E)

    return embeddings


def normalize_distance(D: np.ndarray, factor: float = 1.5) -> np.ndarray:
    D = np.array(D)
    finite_vals = D[np.isfinite(D) & (D > 0)]

    penalty = np.percentile(finite_vals, 95) * factor if len(finite_vals) > 0 else 1.0
    D[np.isinf(D)] = penalty
    np.fill_diagonal(D, 0)
    return D


def compute_structure_distances(graphs: List[nx.Graph], factor: float = 1.5) -> List[np.ndarray]:
    dist: List[np.ndarray] = [np.array(nx.floyd_warshall_numpy(G)) for G in graphs]

    for i, G in enumerate(graphs):
        dist[i] = normalize_distance(dist[i], factor=factor)

    return dist


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
    T, log = ot.gromov.fused_gromov_wasserstein(M, D1, D2, p, q, alpha=alpha, log=True)
    return log['fgw_dist'], T


# Algorithme de fusion de graphe sous couvert de distance FGW

    





def compute_node_features(G, embeddings):
    return np.array([embeddings[n] for n in G.nodes])

def compute_structure_matrix(G):
    return np.array(nx.floyd_warshall_numpy(G))

def recursive_traversal_with_scores_fusion_fgw_base(
    start_node: str,
    question: str,
    embeddings: EmbeddingStore,
    G: nx.DiGraph,
    model: SentenceTransformer,
    alpha: float = 0.5,
    max_depth: int = 3,
    sim_threshold: float = 0.8,
    fgw_threshold: float = 0.2
) -> List[TraversalStep]:

    path: List[TraversalStep] = [TraversalStep(node_id=start_node, similarity=None)]
    current = start_node
    query_vec = model.encode([question], normalize_embeddings=True)[0]

    for depth in range(max_depth):
        children = list(G.successors(current))
        if not children:
            break

        child_vecs = [embeddings.vectors[c] for c in children]
        sims = np.dot(child_vecs, query_vec)
        max_sim = np.max(sims)
        similar_nodes = [i for i, sim in enumerate(sims) if sim > sim_threshold * max_sim]

        if len(similar_nodes) > 1:
            # Copie du graphe
            G_fus = deepcopy(G)
            embeddings_fus = deepcopy(embeddings.vectors)

            # Construction du nœud fusionné temporaire
            new_node_id = f"fused_{current}_{depth}"
            fused_texts = " | ".join([G_fus.nodes[children[i]]["text"] for i in similar_nodes])
            G_fus.add_node(new_node_id, text=fused_texts, type="fused")
            print("debug0")
            embeddings_fus[new_node_id] = model.encode([fused_texts], normalize_embeddings=True)[0]
            print("debug1")

            # Reconnexion temporaire (attention on essaye de tout fus à la fois alors qu'on pourrait faire un algo glouton peut être à modifier) 
            for i in similar_nodes:
                sim_child = children[i]
                grand_children = list(G_fus.successors(sim_child))
                for gc in grand_children:
                    G_fus.add_edge(new_node_id, gc)
                G_fus.remove_edge(current, sim_child)
                G_fus.remove_node(sim_child)

            G_fus.add_edge(current, new_node_id)

            # Calcul de la distance FGW
            F1,F2 = compute_embeddings([G, G_fus], model=model)
            D1,D2 = compute_structure_distances([G, G_fus])

            fgw_dist, _ = compute_fgw(F1, D1, F2, D2, alpha=alpha)
            print(f"Distance FGW : {fgw_dist}") 

            if fgw_dist < fgw_threshold:
                # Accept fusion
                for i in similar_nodes:
                    sim_child = children[i]
                    grand_children = list(G.successors(sim_child))
                    for gc in grand_children:
                        G.add_edge(new_node_id, gc)
                    G.remove_edge(current, sim_child)
                    G.remove_node(sim_child)

                G.add_node(new_node_id, text=fused_texts, type="fused")
                embeddings.vectors[new_node_id] = embeddings_fus[new_node_id]
                G.add_edge(current, new_node_id)

                current = new_node_id
                mean_score = float(np.mean([sims[i] for i in similar_nodes]))
                path.append(TraversalStep(node_id=current, similarity=mean_score))
                continue  # passe à l'itération suivante

        # Si pas de fusion ou fusion rejetée, suivre meilleur enfant
        best_idx = int(np.argmax(sims))
        current = children[best_idx]
        sim_score = float(sims[best_idx])
        path.append(TraversalStep(node_id=current, similarity=sim_score))

    return path


def recursive_traversal_with_scores_fusion_fgw_enhanced( # Atm : Rajout d'arrêtes mais directement (proba d'existence = 1 ) 
    start_node: str,
    question: str,
    embeddings: EmbeddingStore,
    G: nx.DiGraph,
    model: SentenceTransformer,
    max_depth: int = 3,
    sim_threshold: float = 0.8,
    fgw_threshold: float = 0.1,
    alpha: float = 0.5
) -> List[TraversalStep]:
    path: List[TraversalStep] = [TraversalStep(node_id=start_node, similarity=None)]
    current = start_node
    query_vec = model.encode([question], normalize_embeddings=True)[0]

    for depth in range(max_depth):
        children = list(G.successors(current))
        if not children:
            break

        child_vecs = [embeddings.vectors[c] for c in children]
        sims = np.dot(child_vecs, query_vec)

        sorted_children = sorted(zip(children, sims), key=lambda x: x[1], reverse=True)
        sorted_children_relevant = [c for c in sorted_children if c[1] > sim_threshold * np.max(sims)]
        print(f"Similarité pour {current} : {[f'{c[0]}: {c[1]:.3f}' for c in sorted_children_relevant]}")

        # Gloutonnement : on tente de fusionner les meilleurs enfants tout en respectant fgw_threshold (ici on peut optim en faisant une dichotomie)
        candidate_set = []
        candidate_vecs = []
        for child_id, sim_score in sorted_children_relevant:
            tmp_set = candidate_set + [child_id]
            tmp_texts = [G.nodes[n]["text"] for n in tmp_set]
            tmp_fused_text = " | ".join(tmp_texts)
            tmp_node_id = f"fused_{current}_{depth}"

            # Crée une copie du graphe pour tester la fusion
            G_fus = G.copy()
            G_fus.add_node(tmp_node_id, text=tmp_fused_text, type="fused")

            # Connexions dans le graphe modifié
            for c in tmp_set:
                for gc in G.successors(c):
                    G_fus.add_edge(tmp_node_id, gc)
                G_fus.remove_edge(current, c)
                G_fus.remove_node(c)
            G_fus.add_edge(current, tmp_node_id)

            # Recalcule embeddings et distances
            Gs = [G, G_fus]
            feats = compute_embeddings(Gs, model=model)
            dists = compute_structure_distances(Gs)
            fgw_dist, _ = compute_fgw(feats[0], dists[0], feats[1], dists[1], alpha=alpha)

            if fgw_dist <= fgw_threshold:
                candidate_set = tmp_set
                candidate_vecs.append(sim_score)
                print(f"Fusion acceptée pour {tmp_set} avec distance FGW : {fgw_dist}")
                distance_restante = fgw_threshold - fgw_dist
            else:
                break  # On arrête dès que la distance devient trop grande
        
        
        if len(candidate_set) > 1:
            fused_text = " | ".join([G.nodes[n]["text"] for n in candidate_set])
            new_node_id = f"fused_{current}_{depth}"
            G.add_node(new_node_id, text=fused_text, type="fused")
            embeddings.vectors[new_node_id] = model.encode([fused_text], normalize_embeddings=True)[0]

            for c in candidate_set:
                for gc in G.successors(c):
                    G.add_edge(new_node_id, gc)
                G.remove_edge(current, c)
                G.remove_node(c)

            G.add_edge(current, new_node_id)
            # Glouton sur la création d'arrêtes entre les enfants des noeuds pertinent restant et le nouveau noeud (idem ici on peut optim en faisant une dichotomie): 
            diff_set = sorted_children_relevant[len(candidate_set):]
            # Calcul de la similarité des nouveaux enfants avec la question
            all_grand_children = []
            for c in diff_set:
                grand_children = list(G.successors(c[0]))
                all_grand_children.extend(grand_children)
            if len(all_grand_children) == 0:
                print("Pas de petits-enfants restants")
                continue
            grand_child_vecs = [embeddings.vectors[c] for c in all_grand_children]
            grand_child_sims = np.dot(grand_child_vecs, query_vec)
            sorted_grand_children = sorted(zip(all_grand_children, grand_child_sims), key=lambda x: x[1], reverse=True)
            sorted_grand_children_relevant = [c for c in sorted_grand_children if c[1] > sim_threshold * np.max(grand_child_sims)]
            print(f"Arrêtes: Similarité pour {current} : {[f'{c[0]}: {c[1]:.3f}' for c in sorted_grand_children_relevant]}")
            #distance_restante = fgw_threshold - fgw_dist
            # On fait un algo glouton sur les enfants restants pour voir si on peut les conneter au nouveau noeud
            candidate_set_arretes = []
            candidate_vecs_arretes = []
            for child_id, sim_score in sorted_grand_children_relevant:
                tmp_set_arretes = candidate_set_arretes + [child_id]
                tmp_texts_arretes = [G.nodes[n]["text"] for n in tmp_set_arretes]
                tmp_fused_text_arretes = " | ".join(tmp_texts_arretes)
                
                # Crée une copie du graphe pour tester la fusion
                G_fus_arretes = G.copy()
                # On Change le contenu de new_node_id dans G_fus_arretes en rajoutant tmp_fused_text_arretes
                G_fus_arretes.nodes[new_node_id]["text"] = G_fus_arretes.nodes[new_node_id]["text"] + " | " + tmp_fused_text_arretes
                # On connecte new_node_id avec les tmp_set_arretes
                for c in tmp_set_arretes:
                    G_fus_arretes.add_edge(new_node_id, c)
                

                # Recalcule embeddings et distances
                Gs_arretes = [G, G_fus_arretes]
                feats_arretes = compute_embeddings(Gs_arretes, model=model)
                dists_arretes = compute_structure_distances(Gs_arretes)
                fgw_dist_arretes, _ = compute_fgw(feats_arretes[0], dists_arretes[0], feats_arretes[1], dists_arretes[1], alpha=alpha)
                print(f"Distance FGW arrêtes : {fgw_dist_arretes}")
                print(f"Distance restante : {distance_restante}")
                if fgw_dist_arretes <= distance_restante:
                    candidate_set_arretes = tmp_set_arretes
                    candidate_vecs_arretes.append(sim_score)
                    print(f"Arretes : Fusion acceptée pour {tmp_set_arretes} avec distance FGW : {fgw_dist_arretes}")
                else:
                    break
            if len(candidate_set_arretes) > 1:
                G.nodes[new_node_id]["text"] = G.nodes[new_node_id]["text"] + " | " + tmp_fused_text_arretes
                for c in candidate_set_arretes:
                    G.add_edge(new_node_id, c)
                    


            current = new_node_id
            path.append(TraversalStep(node_id=current, similarity=float(np.mean(candidate_vecs))))

        else:
            best_idx = int(np.argmax(sims))
            current = children[best_idx]
            path.append(TraversalStep(node_id=current, similarity=float(sims[best_idx])))

    return path

