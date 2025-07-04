import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances
import ot
from copy import deepcopy
import uuid
from kt_gen.knowledge_graph.utils.pydantic_models import EmbeddingStore, TraversalStep, Node, GraphWrapper, EmbeddingMatrix
from kt_gen.knowledge_graph.utils.utils_fgw import compute_embeddings, compute_structure_distances, compute_fgw, elementary_distance, compute_gw, compute_similarity_weighted_structure_distances
import time 
import sys
np.set_printoptions(threshold=sys.maxsize)




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
    embedding_current_graph = compute_embeddings([G], model=model)
    dist_current_graph = compute_structure_distances([G])

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
            F2 = compute_embeddings([G_fus], model=model)
            D2 = compute_structure_distances([G_fus])

            fgw_dist, _ = compute_fgw(embedding_current_graph[0], dist_current_graph[0], F2, D2, alpha=alpha)
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
    alpha: float = 0.5,
    accelerated: bool = True
) -> List[TraversalStep]:
    path: List[TraversalStep] = [TraversalStep(node_id=start_node, similarity=None)]
    current = start_node
    query_vec = model.encode([question], normalize_embeddings=True)[0]
    distance_restante = fgw_threshold
    embeddings_current_graph = compute_embeddings([G], model=model)
    dist_current_graph = compute_structure_distances([G])
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
        G_fus = G.copy()

        for child_id, sim_score in sorted_children_relevant:
            tmp_set = candidate_set + [child_id]
            #for elm in tmp_set:
              #print(elm, G.nodes[elm])
            tmp_texts = [G.nodes[n]["text"] for n in tmp_set]
            tmp_fused_text = " | ".join(tmp_texts)
            tmp_node_id = f"fused_{current}_{depth}_{uuid.uuid4().hex[:8]}"
            # Check if tmp_node_id already exists
            if tmp_node_id in G.nodes:
                # Modify the text of the existing node
                G_fus.nodes[tmp_node_id]["text"] = tmp_fused_text 
            

            # Crée une copie du graphe pour tester la fusion
            #G_fus = G.copy()
            G_fus.add_node(tmp_node_id, text=tmp_fused_text, type="fused")

            # Connexions dans le graphe modifié
            """
            for c in tmp_set:
                for gc in G.successors(c):
                    G_fus.add_edge(tmp_node_id, gc)
                G_fus.remove_edge(current, c)
                G_fus.remove_node(c)
            G_fus.add_edge(current, tmp_node_id)
            """
            for gc in G.successors(child_id):
                G_fus.add_edge(tmp_node_id, gc)
            G_fus.remove_edge(current, child_id)

            G_fus.remove_node(child_id)


            G_fus.add_edge(current, tmp_node_id)
            #print("*-----------------************************************------------------****")
            #print(current, child_id, tmp_node_id)




            # Recalcule embeddings et distances
            Gs = [G_fus]
            feats = compute_embeddings(Gs, model=model) # ICI potentielle grosse optimisation à faire si on se base uniquement sur dists
            dists = compute_structure_distances(Gs)
            if accelerated : 
                fgw_dist, _ = compute_gw(dist_current_graph[0], dists[0])
            else:
                fgw_dist, _ = compute_fgw(embeddings_current_graph[0], dist_current_graph[0], feats[0], dists[0], alpha=alpha)
            print(f"FGW dist : {fgw_dist}")
            if fgw_dist <= distance_restante:
                candidate_set = tmp_set
                candidate_vecs.append(sim_score)
                print(f"Fusion acceptée pour {tmp_set} avec distance FGW : {fgw_dist}")
                distance_restante = fgw_threshold - fgw_dist
            else:
                break  # On arrête dès que la distance devient trop grande
        
        
        if len(candidate_set) > 1:
            fused_text = " | ".join([G.nodes[n]["text"] for n in candidate_set])
            # Generate random ID for the new node
            new_node_id = f"fused_{current}_{depth}_{uuid.uuid4().hex[:8]}"
            G.add_node(new_node_id, text=fused_text, type="fused")
            embeddings.vectors[new_node_id] = model.encode([fused_text], normalize_embeddings=True)[0]

            for c in candidate_set:
                for gc in G.successors(c):
                    G.add_edge(new_node_id, gc)
                G.remove_edge(current, c)
                G.remove_node(c)

            G.add_edge(current, new_node_id)
            # Glouton sur la création d'arrêtes entre les enfants des noeuds pertinent restant et le nouveau noeud (idem ici on peut optim en faisant une dichotomie): 
            embeddings_current_graph = compute_embeddings([G], model=model)
            dist_current_graph = compute_structure_distances([G])
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
            G_fus_arretes = G.copy()
            tmp_texts_arretes = ""
            for child_id, sim_score in sorted_grand_children_relevant:
                tmp_set_arretes = candidate_set_arretes + [child_id]
                tmp_texts_arretes = tmp_texts_arretes + " | " + G.nodes[child_id]["text"]
                
                # Crée une copie du graphe pour tester la fusion
                #G_fus_arretes = G.copy()
                # On Change le contenu de new_node_id dans G_fus_arretes en rajoutant tmp_fused_text_arretes
                G_fus_arretes.nodes[new_node_id]["text"] = G_fus_arretes.nodes[new_node_id]["text"] + " | " + G.nodes[child_id]["text"]
                # On connecte new_node_id avec les tmp_set_arretes
                #for c in tmp_set_arretes:
                #    G_fus_arretes.add_edge(new_node_id, c)
                G_fus_arretes.add_edge(new_node_id, child_id)

                # Recalcule embeddings et distances
                Gs_arretes = [G_fus_arretes]
                feats_arretes = compute_embeddings(Gs_arretes, model=model)
                dists_arretes = compute_structure_distances(Gs_arretes)
                fgw_dist_arretes, _ = compute_fgw(embeddings_current_graph[0], dist_current_graph[0], feats_arretes[0], dists_arretes[0], alpha=alpha)
                print(f"Distance FGW arrêtes : {fgw_dist_arretes}")
                print(f"Distance restante : {distance_restante}")
                if fgw_dist_arretes <= distance_restante:
                    candidate_set_arretes = tmp_set_arretes
                    candidate_vecs_arretes.append(sim_score)
                    print(f"Arretes : Fusion acceptée pour {tmp_set_arretes} avec distance FGW : {fgw_dist_arretes}")
                else:
                    break
            if len(candidate_set_arretes) > 1:
                G.nodes[new_node_id]["text"] = G.nodes[new_node_id]["text"]  + tmp_texts_arretes
                for c in candidate_set_arretes:
                    G.add_edge(new_node_id, c)
                    


            current = new_node_id
            path.append(TraversalStep(node_id=current, similarity=float(np.mean(candidate_vecs))))

        else:
            best_idx = int(np.argmax(sims))
            current = children[best_idx]
            path.append(TraversalStep(node_id=current, similarity=float(sims[best_idx])))

    return path


# Fonction de répartition d'un "score de récompense" en fonction de la similarité

def reward_distribution(Edges_Similarity: Dict[Tuple[str, str], float], reward: float, lr: float = 0.8) -> Optional[Dict[Tuple[str, str], float]]:
    """ 
    Fonction qui prend en entrée un dictionnaire de similarité des arêtes et une récompense.
    La fonction va répartir le reward en fonction de la similarité des arêtes (les arêtes ayant plus que la moyenne se voient attribuer un reward positif dépendant de l'écart à la moyenne, les autres un reward négatif).
    """
    if not Edges_Similarity:
        return None
    
    mean_similarity = np.mean(list(Edges_Similarity.values()))
    if mean_similarity == 0:
        return None  # Pas de similarité, pas de récompense
    updated_edges = {}
    # Reward doit être réparti en pourcentage (ie la somme des changements à la fin doit être égale à 0 et en valeur absolue à 2* reward)
    for edge, similarity in Edges_Similarity.items():
        if similarity >= mean_similarity:
            change = lr * (similarity - mean_similarity) * reward
            """ Je commente la partie de régulation pour observer l'incrémentation de la proba
        else:
            change = -lr * (mean_similarity - similarity) * reward
            print(f"Je me fais casser la gueule pour l'arête {edge} avec une similarité de {similarity:.3f} et un changement de {change:.3f}")
            """
        updated_edges[edge] = change  





    return updated_edges







def recursive_traversal_with_scores_fusion_fgw_genetic(
    start_node: str,
    question: str,
    embeddings: EmbeddingStore,
    G: nx.DiGraph,
    model: SentenceTransformer,
    max_depth: int = 3,
    sim_threshold: float = 0.8,
    fgw_threshold: float = 0.1,
    alpha: float = 0.5,
    initial_prob: float = 0.8,
    initial_prob_new: float = 0.1,
    lr: float = 0.8,
    structure_distance: str = "base"
) -> List[TraversalStep]:

    path: List[TraversalStep] = [TraversalStep(node_id=start_node, similarity=None)]
    time_start = time.time()
    current = start_node
    query_vec = model.encode([question], normalize_embeddings=True)[0]
    distance_restante = fgw_threshold
    embedding_current_graph = compute_embeddings([G])
    if structure_distance == "similarity_weighted":
        dists_current_graph = compute_similarity_weighted_structure_distances([G], embedding_current_graph)
    else:
        dists_current_graph = compute_structure_distances([G])
    temps_wasted = 0
    for depth in range(max_depth):
        children = list(G.successors(current))
        if not children:
            break

        child_vecs = [embeddings.vectors[c] for c in children]
        sims = np.dot(child_vecs, query_vec)

        sorted_children = sorted(zip(children, sims), key=lambda x: x[1], reverse=True)
        sorted_children_relevant = [c for c in sorted_children if c[1] > sim_threshold * np.max(sims)]
        sorted_children_proba = [c for c in sorted_children if c[1] * G.edges[current, c[0]].get("prob", 1.0) > sim_threshold * np.max(sims)]
        diff_prob= [c for c in sorted_children_proba if c not in sorted_children_relevant]
        candidate_set = []
        candidate_vecs = []
        G_fus = G.copy()
        print(f"Temps préprocessus : {time.time() - time_start:.3f} secondes")

        for child_id, sim_score in sorted_children_proba:
            #print(f"Enfant : {child_id}, Similarité : {sim_score}")
            tmp_set = candidate_set + [child_id]
            tmp_texts = [G.nodes[n]["text"] for n in tmp_set]
            tmp_fused_text = " | ".join(tmp_texts)
            tmp_node_id = f"fused_{current}_{depth}_{uuid.uuid4().hex[:8]}"

            if tmp_node_id in G.nodes:
                G_fus.nodes[tmp_node_id]["text"] = tmp_fused_text
            else:
                G_fus.add_node(tmp_node_id, text=tmp_fused_text, type="fused")
            for gc in G.successors(child_id):
                G_fus.add_edge(tmp_node_id, gc, prob=G.edges[child_id, gc].get("prob", initial_prob_new), structure_distance=G.edges[child_id, gc].get("structure_distance", 1.0))
            # La nouvelle proba étant la moyenne de toutes les probas des enfants fusionnés, same pour la distance de structure

            new_prob = np.mean([G.edges[child_id, gc].get("prob", initial_prob_new) for gc in G.successors(child_id)])
            new_structure_distance = np.mean([G.edges[child_id, gc].get("structure_distance", 1.0) for gc in G.successors(child_id)])
            G_fus.add_edge(current, tmp_node_id, prob=new_prob, structure_distance=new_structure_distance)
            G_fus.remove_edge(current, child_id)
            G_fus.remove_node(child_id)
            time_start = time.time()
            Gs = [G_fus]  # Là je peux optim en évitant de recalculer sur le premier graphe

            feats = compute_embeddings(Gs, model=model)
            #print(f"Temps de calcul des embeddings : {time.time() - time_start:.3f} secondes")
            temps_wasted += time.time() - time_start
            if structure_distance == "similarity_weighted":
                time_start = time.time()
                dists = compute_similarity_weighted_structure_distances(Gs, feats) # Idem ici 
                #print(f"Temps de calcul des distances structurelles pondérées : {time.time() - time_start:.3f} secondes")
                temps_wasted += time.time() - time_start
            else:
                dists = compute_structure_distances(Gs) # Idem ici
            fgw_dist, _ = compute_fgw(embedding_current_graph[0], dists_current_graph[0], feats[0], dists[0], alpha=alpha)

            if fgw_dist <= distance_restante:
                candidate_set = tmp_set
                candidate_vecs.append(sim_score)
                print(f"Fusion acceptée pour {tmp_set} avec distance FGW : {fgw_dist}")

                distance_restante = fgw_threshold - fgw_dist
            else:
                break

        if len(candidate_set) > 1:
            time_start = time.time()
            fused_text = " | ".join([G.nodes[n]["text"] for n in candidate_set])
            new_node_id = f"fused_{current}_{depth}_{uuid.uuid4().hex[:8]}"
            G.add_node(new_node_id, text=fused_text, type="fused")
            embeddings.vectors[new_node_id] = model.encode([fused_text], normalize_embeddings=True)[0]




            new_prob = np.mean([G.edges[current, c].get("prob", initial_prob_new) for c in candidate_set ])
            new_structure_distance = np.mean([G.edges[current, c].get("structure_distance", 1.0) for c in candidate_set])
            G.add_edge(current, new_node_id, prob= new_prob, structure_distance=new_structure_distance)
            #print(f"Ajout d'une arrête de {current} vers {new_node_id} avec probabilité {initial_prob:.3f}")
            for c in candidate_set:
                
                for gc in G.successors(c):
                    G.add_edge(new_node_id, gc, prob= G.edges[c,gc].get("prob", initial_prob_new), structure_distance=G.edges[c,gc].get("structure_distance", 1.0))
                G.remove_edge(current, c)
                G.remove_node(c)
            # Glouton sur ajout d'arêtes pondérées
            embedding_current_graph = compute_embeddings([G], model=model)
            print(f"Temps de recalcul des embeddings + fusion : {time.time() - time_start:.3f} secondes")
            if structure_distance == "similarity_weighted":
                dists_current_graph = compute_similarity_weighted_structure_distances([G], embedding_current_graph)
            else:
                dists_current_graph = compute_structure_distances([G])

            
            diff_set = sorted_children_proba[len(candidate_set):] + diff_prob
            #print("candidate_set", candidate_set)
            #print("sorted_children_relevant : ", sorted_children_relevant)
            #print("diff_set : ", diff_set)


            all_grand_children = []
            edges_diff_set = []
            for c in diff_set:
                grand_children = list(G.successors(c[0]))
                all_grand_children.extend([(c[0], gc) for gc in grand_children])  # On garde le parent
                edges_diff_set.extend([(c[0], gc) for gc in grand_children])

            if len(all_grand_children) == 0:
                print("Pas de petits-enfants restants")
                continue
            #print(f"all_grand_children : {[f'{gc[0]} -> {gc[1]}' for gc in all_grand_children]}")
            #print("edges_diff_set : ", edges_diff_set)

            if len(all_grand_children) > 0:
                grand_child_vecs = [embeddings.vectors[gc[1]] for gc in all_grand_children]  # gc[1] = id du petit-enfant
                grand_child_sims = np.dot(grand_child_vecs, query_vec)
                sorted_gc = sorted(zip(all_grand_children, grand_child_sims), key=lambda x: x[1], reverse=True)
                sorted_gc_relevant = [c for c in sorted_gc if c[1] > sim_threshold * np.max(grand_child_sims)]
                sorted_gc_proba = [c for c in sorted_gc if c[1] * G.edges[c[0][0], c[0][1]].get("prob", 0.1) > sim_threshold * np.max(grand_child_sims)]
                #print("sorted_gc_proba : ", sorted_gc_proba)
                diff_prob = [c for c in sorted_gc_proba if c not in sorted_gc_relevant]
                #print(f"Noeuds de diff_prob: {[f'{c[0][1]}: {c[1]:.3f} {c[1]*G.edges[c[0][0], c[0][1]].get('prob', 1.0)}' for c in diff_prob]}")
                #print(f"Noeuds enfats considérés avec score associé : {[f'{c[0][1]}: {c[1]:.3f} {c[1]*G.edges[c[0][0], c[0][1]].get('prob', 1.0)}' for c in sorted_gc_relevant]}")
                candidate_set_arretes = []
                candidate_vecs_arretes = []
                G_fus_arretes = G.copy()
                tmp_texts_arretes = ""
                for child_id, sim_score in sorted_gc_proba:
                    #print(f"Enfant : {child_id[1]}, Similarité : {sim_score}")
                    tmp_set_arretes = candidate_set_arretes + [child_id[1]]
                    tmp_texts_arretes = tmp_texts_arretes + " | " + G.nodes[child_id[1]]["text"]
                    G_fus_arretes.nodes[new_node_id]["text"] = G_fus_arretes.nodes[new_node_id]["text"] + " | " + G.nodes[child_id[1]]["text"]
                    G_fus_arretes.add_edge(new_node_id, child_id[1], prob=G.edges[child_id[0], child_id[1]].get("prob", initial_prob_new), structure_distance=G.edges[child_id[0], child_id[1]].get("structure_distance", 1.0))

                    Gs_arretes = [G_fus_arretes] # Idem ça sert à rien de recalculer sur le premier graphe
                    time_start = time.time()
                    feats_arretes = compute_embeddings(Gs_arretes, model=model)
                    temps_wasted += time.time() - time_start

                    if structure_distance == "similarity_weighted":
                        time_start = time.time()
                        dists_arretes = compute_similarity_weighted_structure_distances(Gs_arretes, feats_arretes) # Idem ça sert à rien de recalculer sur le premier graphe
                        temps_wasted += time.time() - time_start
                    else:
                        dists_arretes = compute_structure_distances(Gs_arretes) # Idem ça sert à rien de recalculer sur le premier graphe
                    fgw_dist_arretes, _ = compute_fgw(embedding_current_graph[0],dists_current_graph[0], feats_arretes[0], dists_arretes[0], alpha=alpha)

                    if fgw_dist_arretes <= distance_restante:
                        candidate_set_arretes = tmp_set_arretes
                        candidate_vecs_arretes.append(sim_score) # Jpeux utiliser ce vec pour transmettre des sim?
                    else:
                        break

                #print(f"Candidate set arretes : {candidate_set_arretes}")
                #print(f"Candidate vecs arretes : {candidate_vecs_arretes}")
                edge_to_consider = {}
                #### !!!!!!!!!!!! Dans toute cette partie FACT CHECK si je recalcule pas des similitudes inutiles !!!!!!!!!!!!!!
                time_start = time.time()
                for child in candidate_set_arretes: 
                    if G.has_edge(new_node_id, child):
                        sim= np.dot(embeddings.vectors[child], query_vec) # Faudra check le produit scalaire si c'est pas négatif
                        edge_to_consider[(new_node_id, child)] = sim 
                    else: 
                        sim = np.dot(embeddings.vectors[child], query_vec)
                        G.add_edge(new_node_id, child, prob=initial_prob_new, structure_distance= 1.0/(sim + 1e-6))  # On évite la division par zéro
                        edge_to_consider[(new_node_id, child)] = sim
                #print(f"Edges to consider : {edge_to_consider}")
                for edge, _ in sorted_gc_relevant + diff_prob:
                    if G.has_edge(new_node_id, edge[1]):
                        sim= np.dot(embeddings.vectors[edge[1]], query_vec)
                        edge_to_consider[(new_node_id, edge[1])] = sim
                    else:
                        sim = np.dot(embeddings.vectors[edge[1]], query_vec)
                        G.add_edge(new_node_id, edge[1], prob=initial_prob_new, structure_distance= 1.0/(sim + 1e-6))  # On évite la division par zéro
                        edge_to_consider[(new_node_id, edge[1])] = sim
                #print(f"Edges to consider : {edge_to_consider}")

                rewarded_edges = reward_distribution(edge_to_consider, reward=5, lr=lr)

                if rewarded_edges:
                    for (u, v), change_prob in rewarded_edges.items():
                        #print(f"Updating edge {u} -> {v} with new probability {change_prob:.3f}")
                        #print(f"Old prob: {G.edges[u, v].get('prob', -1.0):.3f}")

                        G.edges[u, v]["prob"] = min(G.edges[u, v]["prob"] + change_prob, 1.0)
                        #print(f"New prob: {G.edges[u, v].get('prob', -1.0):.3f}")
                print(f"Temps de recalcul des arrêtes : {time.time() - time_start:.3f} secondes")

            current = new_node_id
            path.append(TraversalStep(node_id=current, similarity=float(np.mean(candidate_vecs))))
            print(path)

        else:
            best_idx = int(np.argmax(sims))
            current = children[best_idx]
            path.append(TraversalStep(node_id=current, similarity=float(sims[best_idx])))
            print(path)
    for u, v, data in G.edges(data=True):
        for key, value in data.items():
            if isinstance(value, (np.float32, np.float64)):
                data[key] = float(value)

    for n, data in G.nodes(data=True):
        for key, value in data.items():
            if isinstance(value, (np.float32, np.float64)):
                data[key] = float(value)
    print(f"Temps total gaspillé dans les calculs : {temps_wasted:.3f} secondes")
    return path





# Test optimisation sans recalculer les embeddings et distances de structure à chaque itération







# Cette fonction est suppposée être la version la plus avancée et la moins stable mais la plus optimisée


def test_fusion(tmp_set, candidate_set, G, G_fus, map_index_node, 
                embedding_current_graph, dists_current_graph, 
                model, current, depth, uuid_fus,
                initial_prob_new, alpha, structure_distance):
    """
    Prépare un graphe avec tmp_set fusionné et calcule la FGW distance.
    Renvoie (fgw_dist, feats, dists, G_fus_copy)
    """
    import copy
    import numpy as np

    new_embeddings_fus = copy.deepcopy(embedding_current_graph)
    new_distance_struct = copy.deepcopy(dists_current_graph)
    G_fus_copy = copy.deepcopy(G_fus)
    map_index_node_copy = copy.deepcopy(map_index_node)

    tmp_texts = [G.nodes[n]["text"] for n in tmp_set]
    tmp_fused_text = " | ".join(tmp_texts)
    tmp_node_id = f"fused_{current}_{depth}_{uuid_fus}"

    # supprimer les anciens nœuds de tmp_set dans embeddings & distances
    for n in tmp_set:
        index_to_del = map_index_node_copy[n]
        new_embeddings_fus[0] = np.delete(new_embeddings_fus[0], index_to_del, axis=0)
        new_distance_struct[0] = np.delete(new_distance_struct[0], index_to_del, axis=0)
        new_distance_struct[0] = np.delete(new_distance_struct[0], index_to_del, axis=1)
        for n2, idx in map_index_node_copy.items():
            if idx > index_to_del:
                map_index_node_copy[n2] -= 1

    # ajouter le nouveau nœud fusionné
    if tmp_node_id in G_fus_copy.nodes:
        G_fus_copy.nodes[tmp_node_id]["text"] = tmp_fused_text
    else:
        G_fus_copy.add_node(tmp_node_id, text=tmp_fused_text, type="fused")
        new_embeddings_fus[0] = np.append(
            new_embeddings_fus[0],
            model.encode([tmp_fused_text], normalize_embeddings=True)[0].reshape(1, -1),
            axis=0
        )
        new_distance_struct[0] = np.append(
            new_distance_struct[0],
            np.full((new_distance_struct[0].shape[0], 1), np.exp(2) * 1.5),
            axis=1
        )
        new_distance_struct[0] = np.append(
            new_distance_struct[0],
            np.full((1, new_distance_struct[0].shape[1]), np.exp(2) * 1.5),
            axis=0
        )
        new_distance_struct[0][-1, -1] = 0.0

    # ajouter les arêtes des enfants fusionnés vers leurs successeurs
    for n in tmp_set:
        for gc in G.successors(n):
            G_fus_copy.add_edge(
                tmp_node_id, gc,
                prob=G.edges[n, gc].get("prob", initial_prob_new),
                structure_distance=G.edges[n, gc].get("structure_distance", 1.0)
            )
            new_distance_struct[0][-1, map_index_node_copy[gc]] = G.edges[n, gc].get("structure_distance", 1.0)

    # ajouter l'arête entre current et tmp_node_id
    new_prob = np.mean([G.edges[current, gc].get("prob", initial_prob_new) for gc in tmp_set])
    new_structure_distance = np.mean([G.edges[current, gc].get("structure_distance", 1.0) for gc in tmp_set])
    G_fus_copy.add_edge(current, tmp_node_id, prob=new_prob, structure_distance=new_structure_distance)
    new_distance_struct[0][map_index_node_copy[current], -1] = new_structure_distance

    # retirer les anciens enfants fusionnés
    for child_id in tmp_set:
        if G_fus_copy.has_edge(current, child_id):
            G_fus_copy.remove_edge(current, child_id)
        if G_fus_copy.has_node(child_id):
            G_fus_copy.remove_node(child_id)

    if structure_distance != "similarity_weighted":
        dists = compute_structure_distances([G_fus_copy])
    else:
        dists = new_distance_struct

    # calculer FGW distance
    fgw_dist, _ = compute_fgw(
        embedding_current_graph[0], dists_current_graph[0],
        new_embeddings_fus[0], dists[0], alpha=alpha
    )

    return fgw_dist, new_embeddings_fus, dists, G_fus_copy, map_index_node_copy


def dichotomie_fusion(sorted_children_proba, candidate_set, candidate_vecs, 
                      G, G_fus, map_index_node,
                      embedding_current_graph, dists_current_graph, 
                      model, current, depth, uuid_fus,
                      initial_prob_new, alpha, structure_distance, 
                      distance_restante):
    left, right = 1, len(sorted_children_proba)
    best_k = 0
    best_candidate_set = candidate_set
    best_candidate_vecs = candidate_vecs[:]
    best_embedding = embedding_current_graph
    best_distance = dists_current_graph
    best_G_fus = G_fus

    while left <= right:
        mid = (left + right) // 2
        tmp_set = candidate_set + [cid for cid, _ in sorted_children_proba[:mid]]
        tmp_vecs = candidate_vecs + [s for _, s in sorted_children_proba[:mid]]

        fgw_dist, new_emb, new_dist, new_G_fus, map_index_node_copy = test_fusion(
            tmp_set, candidate_set, G, G_fus, map_index_node,
            embedding_current_graph, dists_current_graph,
            model, current, depth, uuid_fus,
            initial_prob_new, alpha, structure_distance
        )

        if fgw_dist <= distance_restante:
            # on peut aller plus loin
            best_k = mid
            best_candidate_set = tmp_set
            best_candidate_vecs = tmp_vecs
            best_embedding = new_emb
            best_distance = new_dist
            best_G_fus = new_G_fus
            left = mid + 1
        else:
            # trop loin, on réduit
            right = mid - 1
    print(f"Ensemble optimal contenant  : {best_candidate_set} avec une distance FGW de {fgw_dist:.3f}")
    print(f"L'ensemble total des candidats est : {sorted_children_proba}")



    return best_candidate_set, best_candidate_vecs, best_embedding, best_distance, best_G_fus, map_index_node_copy


def recursive_traversal_with_score_fusion_fgw_genetic_optimized(
    start_node: str,
    question: str,
    embeddings: EmbeddingStore,
    G: nx.DiGraph,
    model: SentenceTransformer,
    max_depth: int = 3,
    sim_threshold: float = 0.8,
    fgw_threshold: float = 0.1,
    alpha: float = 0.5,
    initial_prob: float = 0.8,
    initial_prob_new: float = 0.1,
    lr: float = 0.8,
    structure_distance: str = "base"
) -> List[TraversalStep]:

    path: List[TraversalStep] = [TraversalStep(node_id=start_node, similarity=None)]
    time_start = time.time()
    current = start_node
    query_vec = model.encode([question], normalize_embeddings=True)[0]
    distance_restante = fgw_threshold
    embedding_current_graph = compute_embeddings([G])
    map_index_node = {n: i for i, n in enumerate(G.nodes)}
    if structure_distance == "similarity_weighted":
        print("Yow")
        dists_current_graph = compute_similarity_weighted_structure_distances([G], embedding_current_graph)
        for edge in G.edges(data=True):
            G.edges[edge[0], edge[1]]["structure_distance"] = dists_current_graph[0][map_index_node[edge[0]], map_index_node[edge[1]]]

    else:
        dists_current_graph = compute_structure_distances([G])
    temps_wasted = 0
    for depth in range(max_depth):
        print(f"On démarre l'étape à la profondeur {depth} avec le noeud {current}")
        print(f"--------------------------------------------------------------------------------------")
        
        children = list(G.successors(current))
        if not children:
            break

        child_vecs = [embeddings.vectors[c] for c in children]
        sims = np.dot(child_vecs, query_vec)

        sorted_children = sorted(zip(children, sims), key=lambda x: x[1], reverse=True)
        sorted_children_relevant = [c for c in sorted_children if c[1] > sim_threshold * np.max(sims)]
        sorted_children_proba = [c for c in sorted_children if c[1] * G.edges[current, c[0]].get("prob", 1.0) > sim_threshold * np.max(sims)]
        diff_prob= [c for c in sorted_children_proba if c not in sorted_children_relevant]
        candidate_set = []
        candidate_vecs = []
        G_fus = G.copy()
        print(f"Temps préprocessus : {time.time() - time_start:.3f} secondes")

        uuid_fus= uuid.uuid4().hex[:8]  # Génération d'un UUID pour les noeuds fusionnés

        new_embedding_fus_etape_precedente = embedding_current_graph.copy()  # Copie de l'embedding du graphe actuel pour la fusion
        new_distance_struct_etape_precedente = dists_current_graph.copy()  # Copie de la distance de structure du graphe actuel pour la fusion
    

        candidate_set, candidate_vecs, new_embedding_fus_etape_precedente, new_distance_struct_etape_precedente, G_fus, map_index_node_copy = dichotomie_fusion(
            sorted_children_proba, candidate_set, candidate_vecs,
            G, G_fus, map_index_node,
            embedding_current_graph, dists_current_graph,
            model, current, depth, uuid_fus,
            initial_prob_new, alpha, structure_distance,
            distance_restante
            )
        
        map_index_node = map_index_node_copy  # Met à jour l'index des noeuds après fusion

        """
        for child_id, sim_score in sorted_children_proba:
            print(f"Enfant : {child_id}, Similarité : {sim_score}")
            new_embeddings_fus = embedding_current_graph.copy()
            new_distance_struct = dists_current_graph.copy()
            tmp_set = candidate_set + [child_id]
            tmp_texts = [G.nodes[n]["text"] for n in tmp_set]
            tmp_fused_text = " | ".join(tmp_texts)
            tmp_node_id = f"fused_{current}_{depth}_{uuid_fus}"



            for n in tmp_set:#ici pas besoin forcement d'une boucle

                index_to_del = map_index_node[n]

                new_embeddings_fus[0] = np.delete(new_embeddings_fus[0], index_to_del, axis=0)
                new_distance_struct[0] = np.delete(new_distance_struct[0], index_to_del, axis=0)
                new_distance_struct[0] = np.delete(new_distance_struct[0], index_to_del, axis=1)
                for n, idx in map_index_node.items():
                    if idx > index_to_del:
                        map_index_node[n] -= 1
                

            


            if tmp_node_id in G.nodes:
                print("indic bugggg")
                G_fus.nodes[tmp_node_id]["text"] = tmp_fused_text
            else:
                G_fus.add_node(tmp_node_id, text=tmp_fused_text, type="fused")
                new_embeddings_fus[0] = np.append(new_embeddings_fus[0], model.encode([tmp_fused_text], normalize_embeddings=True)[0].reshape(1, -1), axis=0)
                new_distance_struct[0] = np.append(new_distance_struct[0], np.full((new_distance_struct[0].shape[0], 1), np.exp(2) *1.5), axis=1)
                new_distance_struct[0] = np.append(new_distance_struct[0], np.full((1, new_distance_struct[0].shape[1]), np.exp(2) * 1.5), axis=0)
                new_distance_struct[0][-1, -1] = 0.0  
                #print(f"New dist struct avant coef first row: {new_distance_struct[0][0, :]}")
                #print(f"new_distance_struct[0]: {new_distance_struct[0]}")

            for n in tmp_set:
                for gc in G.successors(n):
                    G_fus.add_edge(tmp_node_id, gc, prob=G.edges[n, gc].get("prob", initial_prob_new), structure_distance=G.edges[n, gc].get("structure_distance", 1.0))
                    new_distance_struct[0][-1, map_index_node[gc]] = G.edges[n, gc].get("structure_distance", 1.0)
                    print(f"Ajout de l'arête {tmp_node_id} -> {gc} avec probabilité {G.edges[n, gc].get('prob', initial_prob_new)} et distance de structure {G.edges[n, gc].get('structure_distance', 1.0)}")
                # La nouvelle proba étant la moyenne de toutes les probas des enfants fusionnés, same pour la distance de structure

            new_prob = np.mean([G.edges[current, gc].get("prob", initial_prob_new) for gc in tmp_set])
            new_structure_distance = np.mean([G.edges[current, gc].get("structure_distance", 1.0) for gc in tmp_set])
            G_fus.add_edge(current, tmp_node_id, prob=new_prob, structure_distance=new_structure_distance)
            print(f"Ajout arrête à indice {map_index_node[current]} de {current} vers {tmp_node_id} avec probabilité {new_prob:.3f} et distance de structure {new_structure_distance:.3f}")
            new_distance_struct[0][map_index_node[current], -1] = new_structure_distance
            G_fus.remove_edge(current, child_id)
            G_fus.remove_node(child_id)
            print(f"Suppression de l'arête {current} -> {child_id} et du noeud {child_id} dans G_fus")
            time_start = time.time()
            
            feats = new_embeddings_fus
            dists = new_distance_struct
            #print(f"dist: {dists[0][-1, :]}")
            if structure_distance != "similarity_weighted":

                dists = compute_structure_distances([G_fus])
            fgw_dist, _ = compute_fgw(embedding_current_graph[0], dists_current_graph[0], feats[0], dists[0], alpha=alpha)


                                      


            if fgw_dist <= distance_restante:
                candidate_set = tmp_set
                candidate_vecs.append(sim_score)
                #print(f"Fusion acceptée pour {tmp_set} avec distance FGW : {fgw_dist}")

                distance_restante = fgw_threshold - fgw_dist
                new_embedding_fus_etape_precedente = feats.copy()  # On met à jour l'embedding du graphe actuel pour la fusion
                new_distance_struct_etape_precedente = dists.copy()  # On met à jour la distance de structure du graphe actuel pour la fusion
            else:
                print(f"Fusion refusée pour {tmp_set} avec distance FGW : {fgw_dist}")
                feats = new_embedding_fus_etape_precedente.copy()  # On garde l'embedding du graphe actuel pour la fusion
                dists = new_distance_struct_etape_precedente.copy()  # On garde la distance de structure du graphe actuel pour la fusion
                break
            """

        if len(candidate_set) > 1:
            time_start = time.time()
            fused_text = " | ".join([G.nodes[n]["text"] for n in candidate_set])
            new_node_id = f"fused_{current}_{depth}_{uuid.uuid4().hex[:8]}"
            G.add_node(new_node_id, text=fused_text, type="fused")
            embeddings.vectors[new_node_id] = model.encode([fused_text], normalize_embeddings=True)[0]



            new_prob = np.mean([G.edges[current, c].get("prob", initial_prob_new) for c in candidate_set ])
            new_structure_distance = np.mean([G.edges[current, c].get("structure_distance", 1.0) for c in candidate_set])
            G.add_edge(current, new_node_id, prob= new_prob, structure_distance=new_structure_distance)
            #print(f"Ajout d'une arrête de {current} vers {new_node_id} avec probabilité {initial_prob:.3f}")
            for c in candidate_set:
                
                for gc in G.successors(c):
                    G.add_edge(new_node_id, gc, prob= G.edges[c,gc].get("prob", initial_prob_new), structure_distance=G.edges[c,gc].get("structure_distance", 1.0))
                G.remove_edge(current, c)
                G.remove_node(c)
                print(f"Suppression de l'arête {current} -> {c} et du noeud {c}")
            # Glouton sur ajout d'arêtes pondérées
            map_index_node[new_node_id] = len(G.nodes) - 1  
            embedding_current_graph = new_embedding_fus_etape_precedente.copy()  # On utilise les embeddings déjà calculés
            print(f"Temps de recalcul des embeddings + fusion : {time.time() - time_start:.3f} secondes")
            if structure_distance == "similarity_weighted":
                dists_current_graph = new_distance_struct_etape_precedente.copy()  # On utilise les distances de structure déjà calculées

            else:
                dists_current_graph = compute_structure_distances([G]) # On a pas optim la distance struct de base #TODO



            
            diff_set = sorted_children_proba[len(candidate_set):] + diff_prob




            all_grand_children = []
            edges_diff_set = []
            for c in diff_set:
                grand_children = list(G.successors(c[0]))
                all_grand_children.extend([(c[0], gc) for gc in grand_children])  # On garde le parent
                edges_diff_set.extend([(c[0], gc) for gc in grand_children])

            if len(all_grand_children) == 0:
                print("Pas de petits-enfants restants")
                current = new_node_id
                print(f"current : {current}")
                path.append(TraversalStep(node_id=current, similarity=float(np.mean(candidate_vecs))))
                print(path)
                continue



            if len(all_grand_children) > 0: #ici on peut cook un peu partout avec les embeddings et les distances # TODO mais ça a été bien avancé


                grand_child_vecs = [embeddings.vectors[gc[1]] for gc in all_grand_children]  # gc[1] = id du petit-enfant
                grand_child_sims = np.dot(grand_child_vecs, query_vec)
                sorted_gc = sorted(zip(all_grand_children, grand_child_sims), key=lambda x: x[1], reverse=True)
                sorted_gc_relevant = [c for c in sorted_gc if c[1] > sim_threshold * np.max(grand_child_sims)]
                sorted_gc_proba = [c for c in sorted_gc if c[1] * G.edges[c[0][0], c[0][1]].get("prob", 0.1) > sim_threshold * np.max(grand_child_sims)]
                #print("sorted_gc_proba : ", sorted_gc_proba)
                diff_prob = [c for c in sorted_gc_proba if c not in sorted_gc_relevant]
                candidate_set_arretes = []
                candidate_vecs_arretes = []
                G_fus_arretes = G.copy()
                tmp_texts_arretes = ""

                new_embedding_fus__grand_child_etape_precedente = embedding_current_graph.copy()  # Copie de l'embedding du graphe actuel pour la fusion
                new_distance_struct__grand_child_etape_precedente = dists_current_graph.copy()  # Copie de la distance de structure du graphe actuel pour la fusion
                for child_id, sim_score in sorted_gc_proba:
                    new_embeddings_fus_grand_child = embedding_current_graph.copy()  # Copie de l'embedding du graphe actuel pour la fusion
                    new_distance_struct_grand_child = dists_current_graph.copy()  # Copie de la distance de structure du graphe actuel pour la fusion

                    #print(f"Enfant : {child_id[1]}, Similarité : {sim_score}")
                    tmp_set_arretes = candidate_set_arretes + [child_id[1]]
                    tmp_texts_arretes = tmp_texts_arretes + " | " + G.nodes[child_id[1]]["text"]
                    G_fus_arretes.nodes[new_node_id]["text"] = G_fus_arretes.nodes[new_node_id]["text"] + " | " + G.nodes[child_id[1]]["text"]
                    G_fus_arretes.add_edge(new_node_id, child_id[1], prob=G.edges[child_id[0], child_id[1]].get("prob", initial_prob_new), structure_distance=G.edges[child_id[0], child_id[1]].get("structure_distance", 1.0))
                    
                    new_embeddings_fus_grand_child[0][map_index_node[new_node_id]] = model.encode([G_fus_arretes.nodes[new_node_id]["text"]], normalize_embeddings=True)[0]
                    new_distance_struct_grand_child[0][map_index_node[new_node_id], map_index_node[child_id[1]]] = G.edges[child_id[0], child_id[1]].get("structure_distance", 1.0)



                    feats_arretes = new_embeddings_fus_grand_child
                    dists_arretes = new_distance_struct_grand_child
                    fgw_dist_arretes, _ = compute_fgw(embedding_current_graph[0],dists_current_graph[0], feats_arretes[0], dists_arretes[0], alpha=alpha)

                    if fgw_dist_arretes <= distance_restante:
                        candidate_set_arretes = tmp_set_arretes
                        candidate_vecs_arretes.append(sim_score) # Jpeux utiliser ce vec pour transmettre des sim?
                        new_embedding_fus__grand_child_etape_precedente = feats_arretes.copy()  # On met à jour l'embedding du graphe actuel pour la fusion
                        new_distance_struct__grand_child_etape_precedente = dists_arretes.copy
                    else:
                        feats_arretes = new_embedding_fus__grand_child_etape_precedente.copy()  # On garde l'embedding du graphe actuel pour la fusion
                        dists_arretes = new_distance_struct__grand_child_etape_precedente.copy
                        break




# Partie ajout des nouvelles arêtes pondérées


                edge_to_consider = {}
                #### !!!!!!!!!!!! Dans toute cette partie FACT CHECK si je recalcule pas des similitudes inutiles !!!!!!!!!!!!!!
                time_start = time.time()
                for child in candidate_set_arretes: 
                    if G.has_edge(new_node_id, child):
                        sim= np.dot(embeddings.vectors[child], query_vec) # Faudra check le produit scalaire si c'est pas négatif
                        edge_to_consider[(new_node_id, child)] = sim 
                    else: 
                        sim = np.dot(embeddings.vectors[child], query_vec)
                        G.add_edge(new_node_id, child, prob=initial_prob_new, structure_distance= 1.0/(sim + 1e-6))  # On évite la division par zéro
                        edge_to_consider[(new_node_id, child)] = sim
                #print(f"Edges to consider : {edge_to_consider}")
                for edge, _ in sorted_gc_relevant + diff_prob:
                    if G.has_edge(new_node_id, edge[1]):
                        sim= np.dot(embeddings.vectors[edge[1]], query_vec)
                        edge_to_consider[(new_node_id, edge[1])] = sim
                    else:
                        sim = np.dot(embeddings.vectors[edge[1]], query_vec)
                        G.add_edge(new_node_id, edge[1], prob=initial_prob_new, structure_distance= 1.0/(sim + 1e-6))  # On évite la division par zéro
                        edge_to_consider[(new_node_id, edge[1])] = sim
                #print(f"Edges to consider : {edge_to_consider}")

                rewarded_edges = reward_distribution(edge_to_consider, reward=5, lr=lr)

                if rewarded_edges:
                    for (u, v), change_prob in rewarded_edges.items():
                        G.edges[u, v]["prob"] = G.edges[u, v]["prob"] + change_prob
                        #print(f"New prob: {G.edges[u, v].get('prob', -1.0):.3f}")
                print(f"Temps de recalcul des arrêtes : {time.time() - time_start:.3f} secondes")

            current = new_node_id
            print(f"current : {current}")
            path.append(TraversalStep(node_id=current, similarity=float(np.mean(candidate_vecs))))
            print(path)

        else:
            best_idx = int(np.argmax(sims))
            print(f"children best_idx: {best_idx}")
            current = children[best_idx]
            path.append(TraversalStep(node_id=current, similarity=float(sims[best_idx])))
            print(path)
    for u, v, data in G.edges(data=True):
        for key, value in data.items():
            if isinstance(value, (np.float32, np.float64)):
                data[key] = float(value)

    for n, data in G.nodes(data=True):
        for key, value in data.items():
            if isinstance(value, (np.float32, np.float64)):
                data[key] = float(value)
    print(f"Temps total gaspillé dans les calculs : {temps_wasted:.3f} secondes")
    return path
    

