import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances
import ot
from copy import deepcopy
import uuid


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



