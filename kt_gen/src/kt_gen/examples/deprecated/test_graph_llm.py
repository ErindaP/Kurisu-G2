import networkx as nx
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
from huggingface_hub import configure_http_backend



def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)



# 1. Création du graphe
G = nx.Graph()

G.add_node("doc1", type="definition", text="La photosynthèse est le processus par lequel les plantes produisent de l'énergie à partir de la lumière.")
G.add_node("doc2", type="concept", text="La chlorophylle est un pigment essentiel à la photosynthèse.")
G.add_node("doc3", type="exemple", text="Les feuilles vertes des arbres réalisent la photosynthèse.")
G.add_node("doc4", type="info", text="La respiration cellulaire consomme de l'oxygène et produit du dioxyde de carbone.")
G.add_edge("doc1", "doc2")
G.add_edge("doc1", "doc3")
G.add_edge("doc1", "doc4")

# 1.5 Charger un graphe depuis un fichier JSON : 

import json
def load_graph_from_json(json_bytes):
    import json
    data = json.loads(json_bytes.decode("utf-8"))
    
    G = nx.DiGraph()
    for node in data['nodes']:
        G.add_node(node['index'], text=node['text'])
    for edge in data['edges']:
        G.add_edge(edge[0], edge[1])
    
    return G






# 2. Embedding des nœuds
model = SentenceTransformer("all-MiniLM-L6-v2")  # ou autre modèle plus grand si besoin

node_ids = list(G.nodes)
texts = [G.nodes[n]["text"] for n in node_ids]
embeddings = model.encode(texts, normalize_embeddings=True)

# 3. Indexation FAISS
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(np.array(embeddings))

# 4. Recherche
def find_top_nodes(query, k=3):
    query_vec = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_vec, k)
    return [node_ids[i] for i in indices[0]]

# 5. Construction du prompt
def build_prompt(question, top_node_ids):
    context = "\n".join([f"{nid}: {G.nodes[nid]['text']}" for nid in top_node_ids])
    return f"""Voici des extraits de documents pertinents  :

{context}

Question : {question}
Réponse :"""

# 6. Appel à un LLM (ex: via Ollama)
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def ask_llm(prompt):
    response = client.chat.completions.create(
        model="phi3.5",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content

# 7. Exemple d’utilisation
if __name__ == "__main__":
    question = "Quel est le rôle de la chlorophylle dans la photosynthèse ?"
    top_nodes = find_top_nodes(question, k=3)
    prompt = build_prompt(question, top_nodes)
    answer = ask_llm(prompt)
    print("Réponse du LLM :\n", answer)
