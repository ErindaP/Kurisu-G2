
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
        for child_id, sim_score in sorted_children_proba:
            #print(f"Enfant : {child_id}, Similarité : {sim_score}, sorted_children_proba : {sorted_children_proba}")
            #print(f"Candidate set : {candidate_set}")
            new_embeddings_fus = embedding_current_graph.copy()
            #new_distance_struct = new_distance_struct_etape_precedente.copy()
            new_distance_struct = dists_current_graph.copy()
            #print(f"new_embeddings_fus : {new_embeddings_fus}") 
            #print(f"Enfant : {child_id}, Similarité : {sim_score}")
            tmp_set = candidate_set + [child_id]
            tmp_texts = [G.nodes[n]["text"] for n in tmp_set]
            tmp_fused_text = " | ".join(tmp_texts)
            tmp_node_id = f"fused_{current}_{depth}_{uuid_fus}"
            # Delete the embeddings of each of tmp_set from new_embeddings_fus using map_index_node
            """
            for n in tmp_set:
                index_to_del = map_index_node[n]
                print(f"Shape copy embed new_embeddings_fus.shape: {len(new_embeddings_fus[0])}")
                print(f"Deleting embedding for node {n} at index {index_to_del}")
                new_embeddings_fus[0] = np.delete(new_embeddings_fus[0], index_to_del, axis=0)
                print(f"Shape after deletion new_embeddings_fus.shape: {len(new_embeddings_fus[0])}")
            """
            #print(f"New distance struct original shape: {new_distance_struct[0].shape}")
            #print(f"New distance struct original: {new_distance_struct[0]}")
            # En fait je peux juste supprimer le child_id de new_embeddings_fus
            for n in tmp_set:#ici pas besoin forcement d'une boucle

                index_to_del = map_index_node[n]
                #print(f"Shape copy embed new_embeddings_fus.shape: {len(new_embeddings_fus[0])}")
                #print(f"Deleting embedding for node {child_id} at index {index_to_del}")
                new_embeddings_fus[0] = np.delete(new_embeddings_fus[0], index_to_del, axis=0)
                new_distance_struct[0] = np.delete(new_distance_struct[0], index_to_del, axis=0)
                new_distance_struct[0] = np.delete(new_distance_struct[0], index_to_del, axis=1)
                #print(f"New distance struct shape: {new_distance_struct[0].shape}")
                #print(f"New distance struct: {new_distance_struct[0]}")
                #print(f"Shape after deletion new_embeddings_fus.shape: {len(new_embeddings_fus[0])}")
                # Décallage des index dans map_index_node (pour ceux qui sont après child_id)
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

            for n in tmp_set:
                for gc in G.successors(n):
                    G_fus.add_edge(tmp_node_id, gc, prob=G.edges[n, gc].get("prob", initial_prob_new), structure_distance=G.edges[n, gc].get("structure_distance", 1.0))
                    new_distance_struct[0][-1, map_index_node[gc]] = G.edges[n, gc].get("structure_distance", 1.0)
                    print(f"Ajout de l'arête {tmp_node_id} -> {gc} avec probabilité {G.edges[n, gc].get('prob', initial_prob_new)} et distance de structure {G.edges[n, gc].get('structure_distance', 1.0)}")
                # La nouvelle proba étant la moyenne de toutes les probas des enfants fusionnés, same pour la distance de structure
            #print(f"Shape new_distance_struct : {new_distance_struct[0].shape}")
            #print(f"new dist struct : {new_distance_struct[0]}")
            new_prob = np.mean([G.edges[child_id, gc].get("prob", initial_prob_new) for gc in G.successors(child_id)])
            new_structure_distance = np.mean([G.edges[child_id, gc].get("structure_distance", 1.0) for gc in G.successors(child_id)])
            G_fus.add_edge(current, tmp_node_id, prob=new_prob, structure_distance=new_structure_distance)
            new_distance_struct[0][map_index_node[current], -1] = new_structure_distance
            G_fus.remove_edge(current, child_id)
            G_fus.remove_node(child_id)
            print(f"Suppression de l'arête {current} -> {child_id} et du noeud {child_id} dans G_fus")
            time_start = time.time()
            """ Partie qu'on veut optim 


            Gs = [G_fus]  # Là je peux optim en évitant de recalculer sur le premier graphe

            feats = compute_embeddings(Gs, model=model)
            #print(f"Temps de calcul des embeddings : {time.time() - time_start:.3f} secondes")
            temps_wasted += time.time() - time_start
            if structure_distance == "similarity_weighted":
                time_start = time.time()
                dists = compute_similarity_weighted_structure_distances(Gs, feats) # Idem ici 
                print(f"dists : {dists[0]}")
                #print(f"Temps de calcul des distances structurelles pondérées : {time.time() - time_start:.3f} secondes")
                temps_wasted += time.time() - time_start
            else:
                dists = compute_structure_distances(Gs) # Idem ici
            fgw_dist, _ = compute_fgw(embedding_current_graph[0], dists_current_graph[0], feats[0], dists[0], alpha=alpha)
            """
            
            feats = new_embeddings_fus
            dists = new_distance_struct
            print(f"dist: {dists[0][-1, :]}")




            dists = compute_similarity_weighted_structure_distances([G_fus], feats)
            print(f"dist original: {dists[0][-1, :]}")


            
            #print(f"Shape copy embed feats.shape: {len(feats[0])}")
            #print(f"Number of nodes in feats: {len(G_fus.nodes)}")
            ## Partie qu'on veut optim
            """
            print("new struct dist last row : ", new_distance_struct[0][-1, :])
            if structure_distance == "similarity_weighted": # ici on peut cook un peu plus
                time_start = time.time()
                print(f"embedding_current_graph avant fus : {len(feats[0])}")
                print(f"Nombre de noeuds dans embedding_current_graph avant fus : {len(G_fus.nodes)}")
                #dists = compute_similarity_weighted_structure_distances([G_fus], feats
                dists = compute_similarity_weighted_structure_distances([G_fus], feats)
                print(f"Dist last row: {dists[0][-1, :]}")
                #print(f"Temps de calcul des distances structurelles pondérées : {time.time() - time_start:.3f} secondes")
                temps_wasted += time.time() - time_start
            """
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
            embedding_current_graph = feats.copy()  # On utilise les embeddings déjà calculés
            print(f"Temps de recalcul des embeddings + fusion : {time.time() - time_start:.3f} secondes")
            if structure_distance == "similarity_weighted":
                dists_current_graph = dists.copy()  # On utilise les distances de structure déjà calculées
            """
            if structure_distance == "similarity_weighted":#ici on peut cook un peu aussi
                time_start = time.time()
                print(f"embedding_current_graph : {len(embedding_current_graph[0])}")
                print(f"Nombre de noeuds dans embedding_current_graph : {len(G.nodes)}")
                dists_current_graph = compute_similarity_weighted_structure_distances([G], embedding_current_graph)
                temps_wasted += time.time() - time_start
            else:
                dists_current_graph = compute_structure_distances([G])
            """
            
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

            if len(all_grand_children) > 0: #ici on peut cook un peu partout avec les embeddings et les distances
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

                        G.edges[u, v]["prob"] = G.edges[u, v]["prob"] + change_prob
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
    
