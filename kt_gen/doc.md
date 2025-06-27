# Documentation KT_Gen :


## Formalisation mathématique

### Notations

Nous travaillons majoritairement avec des graphes de connaissances :


- $G = (V, E)$ : Graphe de connaissance dirigé
- $V$ : Ensemble de noeuds
- $E$ : Ensemble d'arêtes
- Soit vertex $v \in V$ $v$ est un couple $(id, text, text_propre)$ avec du contenu textuel 
- $id$ : Identifiant unique du noeud
- $text$ : Contenu textuel du noeud
- $text_propre$ : Contenu textuel caractérisant le noeud 

Sur les graphes nous définissons les ensembles : 
$\mathcal{N}(v)$ : Ensemble des noeuds voisins de $v$ dans le graphe $G$.
$\mathcal{N}(v) = \{u \in V | (v, u) \in E\}$

Sur les graphes de connaissances nous définissons les opérateurs suivants :
$\mathcal{C}(v)$ : L'opérateur renvoyant le contenu textuel du noeud $v$.
$\mathcal{C}(v) = text$

Nous définissons également l'opérateur $\bar{\mathcal{C}}(v)$ : L'opérateur renvoyant le contenu textuel propre du noeud $v$.
$\bar{\mathcal{C}} = text\_propre$


Pour la génération des graphes de connaissances, nous utilisons une structure d'inclusion structurelle définie par : 
$\mathcal{C}(v) = \bar{\mathcal{C}}(v) + \cup_{u \in \mathcal{N}(v)} \mathcal{C}(u)$

### Opérateur de fusion

Nous définissons l'opérateur de fusion $\text{Fus}$ sur les graphes de connaissances comme suit :

$\mathcal{C}_\text{Fus}(\mathbb{V}) = \mathcal{C}(v) + \sum_{u \in \mathbb{F}} \mathcal{C}(u)$


De même, $\mathcal{N}_\text{Fus}(\mathbb{V}) = \bigcup_{u \in \mathbb{F}} \mathcal{N}(u)$

# Algorithme de fusion 

## Version 1 : 
- Parcours en profondeur d'un graphe de connaissance avec comme critère la similbegarité entre la question et le contenu textuel du noeud
- On fusionne les noeuds qui ont une similarité supérieure à un seuil défini ($\epsilon = \text{seuil} \cdot \text{similarite}_{max}$)
- On renvoie le graphe de connaissance fusionné et on continue récursivement le parcours 


## Version 2 :
- Même principe que la version 1 MAIS on n'accepte pas de fusions qui déforme généralement trop le graphe de connaissance (critère avec la distance Fused Gromov Wasserstein) , tant que la fusion déforme trop, on fait un parcours en profondeur des arbres à fusioner jusqu'à ce la distance soit inférieure à un seuil défini. 



-------

## Etat du code : (pour les améliorations/ les maintenances à faire) :

   - Tout ce qui est dans des dossiers `utils` est a priori bon pour le moment et n'est pas supposé être modifié dans l'immediat.
   - Les premières versions des algorithmes sont dans le fichier `kg_base_algo.py` ce sont a priori des algorithmes qu'on aura plus jamais besoin de modifier et qui sont déjà dépréciés.
   - Les algorithmes de fusion utilisant la distance Fused Gromov Wasserstein sont dans le fichier `kg_fgw.py` 
        - La fonction `recursive_traversal_with_scores_fusion_fgw_base` est la version de base de l'algorithme de fusion utilisant la distance Fused Gromov Wasserstein elle essaye de faire que de la fusion de noeuds en essayant de fusionner tous les noeuds qui ont une similarité supérieure à un seuil défini, ainsi si la déformation est plus grande que le seuil autorisé rien ne se passera. C'est la brique de base des algorithmes rafinés développés après mais elle n'est pas destinée à être utilisée directement.


        - La fonction `recursive_traversal_with_scores_fusion_fgw_enhanced` est la première version gloutonne avec de l'ajout d'arête de proba d'existance 1 (pb converge assez rapidement vers graphe complet) mais les fonctionnalités sont a priori ok si on regarde pas l'optimisation ...  
        
        
        - Au sein de ce fichier, la principale fonction en développement est `recursive_traversal_with_scores_fusion_fgw_genetic` qui est la version génétique utilisant la distance Fused Gromov Wasserstein pour fusionner les graphes de connaissances.
            - Cette fonction est en cours de dev et d'optimisation : en particulier il y a probablement beaucoup à faire sur l'optimisation des embeddings qui sont calculés assez souvent et qui sont coûteux en temps de calcul.
            - Il y a aussi des modifications à faire sur les changements de probabilités des arêtes, pour le moment j'utilise un algo qui se base sur la similarité mais qui tend à vouloir équilibrer les modifications ce qui fait que certaines arêtes sont modifiées de manière trop peu importante.
            - Le "glouton" de cet algorithme est pas hyper efficace car on trie par similarité et on break dès qu'une modification est "trop importante" au sens du graphe, le problème est qu'on aurait pourtant pu fusionner d'autres noeuds qui auraient moins déformés : **C'est une problématique assez importante à résoudre**. (Ce phénomène est tempéré par le fait que l'on crée quand même des arêtes dans ce cas là pour les relier)
            



## Dans les choses à faire : 

- **Transformer l'espace utilisé actuellement en rajoutant une distance entre les noeuds (en gros ça revient à dire que les arêtes ça correspond plus à une distance de 1 mais genre plus les contenus sont éloignés plus on veut que la distance soit grande POUR le graphe)** 

- Se renseigner sur notre espace des questions etc cf cahier




        
