import streamlit as st
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import ot
import requests
from huggingface_hub import configure_http_backend
from sklearn.metrics import pairwise_distances

### ATTENTION Il y a une erreur quand alpha vaut 1 (sur tests on devrait avoir des distances nulles mais on a 1 à la place) 


def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)



# 1. Génération de graphes exemples
#@st.cache_data
def generate_graphs():
    G1 = nx.Graph()
    G1.add_nodes_from([
        (0, {"text": "Artificial intelligence is transforming the world"}),
        (1, {"text": "Neural networks learn from data"}),
        (2, {"text": "Deep learning is a subset of machine learning"}),
        (3, {"text": "Data is the new oil"}),
        (4, {"text": "Python is a programming language widely used in AI"}),
        (5, {"text": "TensorFlow and PyTorch are popular frameworks"}),
    ])
    G1.add_edges_from([(0,1), (1,2), (2,3), (3,0), (2,4), (4,5)])

    G2 = nx.Graph()
    G2.add_nodes_from([
        (0, {"text": "Climate change is a major global concern"}),
        (1, {"text": "Carbon emissions contribute to global warming"}),
        (2, {"text": "Renewable energy sources include solar and wind"}),
        (3, {"text": "Fossil fuels are non-renewable"}),
        (4, {"text": "Green technology is evolving rapidly"}),
        (5, {"text": "Government policies can affect climate outcomes"}),
    ])
    G2.add_edges_from([(0,1), (1,2), (2,3), (3,0), (2,4), (4,5)])

    G3 = nx.Graph()
    G3.add_nodes_from([
        (0, {"text": "C++ is used in Artificial Intelligence"}),
        (1, {"text": "Hamlet is a tragedy"}),
        (2, {"text": "Neural networks learn from data"}),
        (3, {"text": "Romeo and Juliet is a love story"}),
        (4, {"text": "C++ is a popular programming language"}),
        (5, {"text": "Literature reflects human emotion"}),
    ])
    G3.add_edges_from([(0,1), (1,2), (2,3), (3,0), (1,4), (4,5)])

    G4 = nx.Graph()
    G4.add_nodes_from([
        (0, {"text": "Python is a programming language"}),
        (1, {"text": "Java is also a programming language"}),
        (2, {"text": "Python is used in data science"}),
        (3, {"text": "Java is used in enterprise applications"}),
        (4, {"text": "Data science involves statistics and programming"}),
    ])

    G4.add_edges_from([(0,1), (1,2), (2,3), (3,0), (2,4)])

    # Graphe orienté cette fois : 
    G5 = nx.DiGraph()
    G5.add_nodes_from([
        (0, {"text": "Artificial intelligence is transforming the world"}),
        (1, {"text": "Neural networks learn from data"}),
        (2, {"text": "Deep learning is a subset of machine learning"}),
        (3, {"text": "Data is the new oil"}),
    ])
    G5.add_edges_from([(0,1), (1,2), (2,3), (3,0)])
    
    # Même graphe pour check si la distance est la même
    G6 = nx.DiGraph()
    G6.add_nodes_from([
        (0, {"text": "Artificial intelligence is transforming the world"}),
        (1, {"text": "Neural networks learn from data"}),
        (2, {"text": "Deep learning is a subset of machine learning"}),
        (3, {"text": "Data is the new oil"}),
    ])
    G6.add_edges_from([(0,1), (1,2), (2,3), (3,0)])

    # Même graphe avec deux arrêtes dans l'autre sens 
    G7 = nx.DiGraph()
    G7.add_nodes_from([
        (0, {"text": "Artificial intelligence is transforming the world"}),
        (1, {"text": "Neural networks learn from data"}),
        (2, {"text": "Deep learning is a subset of machine learning"}),
        (3, {"text": "Data is the new oil"}),
    ])
    G7.add_edges_from([(0,1), (2,1), (3,2), (3,1)])


    return [G1, G2, G3, G4, G5, G6, G7]

#@st.cache_data
def compute_embeddings(_graphs):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []
    for G in graphs:
        texts = [G.nodes[n]["text"] for n in G.nodes()]
        E = model.encode(texts, normalize_embeddings=True)
        embeddings.append(E)
    return embeddings


def normalize_distance(D, factor=1.5):
    D = np.array(D)
    finite_vals = D[np.isfinite(D) & (D > 0)]
    if len(finite_vals) == 0:
        penalty = 1.0
    else:
        penalty = np.percentile(finite_vals, 95) * factor
    D[np.isinf(D)] = penalty
    np.fill_diagonal(D, 0)
    return D
def compute_structure_distances(graphs, factor=1.5):
    dist = [np.array(nx.floyd_warshall_numpy(G)) for G in graphs]

    for i, G in enumerate(graphs):

        # Normaliser les distances
        dist[i] = normalize_distance(dist[i], factor=factor)


        
    # debug
    for elm in dist:
        print(elm)
    return dist
    

def compute_fgw(F1, D1, F2, D2, alpha=0.5):
    p = np.ones(F1.shape[0]) / F1.shape[0]
    q = np.ones(F2.shape[0]) / F2.shape[0]
    M = pairwise_distances(F1, F2, metric="euclidean")
    T, log = ot.gromov.fused_gromov_wasserstein(M, D1, D2, p, q, alpha=alpha, log=True)
    return log['fgw_dist'], T



# 0.5 Import d'un graphe depuis un fichier JSON : 
# Structure du JSON : dictionnaire avec : 
# nodes: List[Dict[str, str]]  # Liste de dictionnaires avec index et texte 
# edges: List[Tuple[str, str]]  # Liste de tuples avec les index des noeuds
def load_graph_from_json(json_bytes):
    import json
    data = json.loads(json_bytes.decode("utf-8"))
    
    G = nx.DiGraph()
    for node in data['nodes']:
        G.add_node(node['index'], text=node['text'])
    for edge in data['edges']:
        G.add_edge(edge[0], edge[1])
    
    return G




# 2. Interface Streamlit
st.set_page_config(page_title="FGW Graph Similarity", layout="centered")
st.title("Comparaison de graphes par Fused Gromov-Wasserstein")

if "graphs" not in st.session_state:
    st.session_state.graphs = []
if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None

graphs = st.session_state.graphs
# Option pour charger un graphe depuis un fichier JSON
uploaded_file = st.file_uploader("Charger un graphe depuis un fichier JSON", type=["json"])


if uploaded_file is not None:
    # Évite d’ajouter plusieurs fois le même fichier
    if uploaded_file.name != st.session_state.last_uploaded_filename:
        try:
            content = uploaded_file.read()
            graph = load_graph_from_json(content)
            st.session_state.graphs.append(graph)
            st.session_state.last_uploaded_filename = uploaded_file.name
            st.success(f"Graphe « {uploaded_file.name} » chargé avec succès.")
        except Exception as e:
            st.error(f"Erreur lors du chargement du graphe : {e}")
    else:
        st.info(f"Le graphe « {uploaded_file.name} » est déjà chargé.")

embeddings = compute_embeddings(graphs)
distances = compute_structure_distances(graphs)

graph_names = [f"Graphe {i+1}" for i in range(len(graphs))]
i = st.selectbox("Sélectionner le premier graphe", options=range(len(graphs)), format_func=lambda x: graph_names[x])
j = st.selectbox("Sélectionner le second graphe", options=range(len(graphs)), index=1, format_func=lambda x: graph_names[x])

if i == j:
    st.warning("Sélectionnez deux graphes différents.")
    st.stop()

alpha = st.slider("Poids alpha (structure vs contenu)", 0.0, 1.0, 0.5, 0.05)

with st.spinner("Calcul de la distance FGW..."):
    st.write("Distances i, j :" , distances[i], distances[j])
    dist, T = compute_fgw(embeddings[i], distances[i], embeddings[j], distances[j], alpha=alpha)

st.success(f"FGW Distance entre {graph_names[i]} et {graph_names[j]} : {dist:.4f}")
#st.write(f'Couplage optimal entre les nœuds : {T}')

# 3. Visualisations
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"{graph_names[i]}")
    fig1, ax1 = plt.subplots()
    pos = nx.spring_layout(graphs[i])
    labels = nx.get_node_attributes(graphs[i], 'text')
    nx.draw(graphs[i], pos, with_labels=True, node_color='skyblue', ax=ax1)
    #nx.draw_networkx_labels(graphs[i], pos, labels=labels, font_size=8, ax=ax1)
    plt.title("Structure du graphe")
    st.pyplot(fig1)

    clicked_node = st.selectbox(f"Selectionner un noeud {graph_names[i]} pour voir son contenu", options=list(graphs[i].nodes))
    st.write(f"Texte du noeud {clicked_node}: {graphs[i].nodes[clicked_node]['text']}")
 
with col2:
    st.subheader(f"{graph_names[j]}")
    fig2, ax2 = plt.subplots()
    pos = nx.spring_layout(graphs[j])
    labels = nx.get_node_attributes(graphs[j], 'text')
    nx.draw(graphs[j], pos, with_labels=True, node_color='lightgreen', ax=ax2)
    #nx.draw_networkx_labels(graphs[j], pos, labels=labels, font_size=8, ax=ax2)
    plt.title("Structure du graphe")
    st.pyplot(fig2)

    clicked_node = st.selectbox(f"Selectionner un noeud  {graph_names[j]} pour voir son contenu", options=list(graphs[j].nodes))
    st.write(f" Texte du noeud {clicked_node}: {graphs[j].nodes[clicked_node]['text']}")

# 4. Couplage optimal
st.subheader("Couplage optimal entre les nœuds (matrice de transport)")
fig3, ax3 = plt.subplots()
sns.heatmap(T, annot=True, fmt=".2f", cmap="Blues", ax=ax3)
ax3.set_xlabel(f"{graph_names[j]} nodes")
ax3.set_ylabel(f"{graph_names[i]} nodes")
st.pyplot(fig3)

# 5. Embedding projection
st.subheader("Projection PCA des embeddings textuels")
pca = PCA(n_components=2)
E1_2D = pca.fit_transform(embeddings[i])
E2_2D = pca.fit_transform(embeddings[j])

fig4, ax4 = plt.subplots()
ax4.scatter(E1_2D[:, 0], E1_2D[:, 1], c='blue', label=graph_names[i])
ax4.scatter(E2_2D[:, 0], E2_2D[:, 1], c='red', label=graph_names[j])

# Add node labels next to the dots
for idx, (x, y) in enumerate(E1_2D):
    ax4.text(x, y, str(idx), color='blue', fontsize=8)
for idx, (x, y) in enumerate(E2_2D):
    ax4.text(x, y, str(idx), color='red', fontsize=8)

ax4.set_title("Embeddings des nœuds (PCA)")
ax4.legend()

st.pyplot(fig4)
# Affichage d'une heatmap avec les distances entre les embeddings
dist_matrix = pairwise_distances(embeddings[i], embeddings[j])
fig5, ax5 = plt.subplots()
sns.heatmap(dist_matrix, annot=True, fmt=".2f", cmap="Blues", ax=ax5)
ax5.set_xlabel(f"{graph_names[j]} nodes")
ax5.set_ylabel(f"{graph_names[i]} nodes")
ax5.set_title("Matrice de distances entre les embeddings")
st.pyplot(fig5)

# Affichage de la dimension de l'espace d'embedding
st.write(f"Dimension de l'espace d'embedding : {embeddings[i].shape[1]}")
