#streamlit: title = "Visualisation Graphique"

import streamlit as st
import json
import networkx as nx
from kt_gen.knowledge_graph.utils.scrud_graph import export_graph_to_json, import_graph_from_json
from kt_gen.utils.streamlit.utils_st import draw_graph, display_graph_pyvis, display_graph_distance, display_graph_distance_with_prob
from kt_gen.utils.llm.utils_llm import resume_context
from openai import OpenAI
# Set the page configuration
st.set_page_config(
    page_title="Graph Visualization",
    page_icon=":bar_chart:",
    layout="wide"
)


# Title of the page
st.title("Graph Visualization")

# File uploader for the graph JSON file
uploaded_file = st.file_uploader("Upload a JSON file containing the graph data", type=["json"])
if uploaded_file is not None:
    try:
        # Load the graph data from the uploaded JSON file
        graph_data = json.load(uploaded_file)
        
        # Import the graph from the JSON data
        graph = import_graph_from_json(graph_data)
        st.session_state["graph"] = graph  
        display_graph_distance_with_prob(graph)
        
        # Option to visualize structure distances with probabilities
        if st.checkbox("Display Structure Distances with Probabilities"):
            display_graph_distance_with_prob(graph, height="600px", width="100%")
    
    except json.JSONDecodeError:
        st.error("The uploaded file is not a valid JSON file.")
    except Exception as e:
        st.error(f"An error occurred while processing the graph: {e}")
else:
    st.info("Please upload a JSON file containing the graph data to visualize it.")



# Button to export the graph to JSON
if st.button("Export Graph to JSON"):
    if 'graph' in locals():
        exported_data = export_graph_to_json(graph)
        st.download_button(
            label="Download Graph as JSON",
            data=json.dumps(exported_data, indent=2),
            file_name="exported_graph.json",
            mime="application/json"
        )
    else:
        st.warning("No graph to export. Please upload a graph first.")
else:
    st.info("Click the button to export the graph to JSON format.")




# Boutton génération axes principaux IA : 
if st.button("Générer les axes principaux IA"):
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    if 'graph' in st.session_state:
        G = st.session_state['graph']
        response_node = {}
        for node in G.nodes:
            if G.nodes[node].get("type") != "fused":
                continue
            context = G.nodes[node]["text"]
            response = resume_context(context, client)
            st.success(f"Réponse pour le nœud {node}: {response}")
            response_node[node] = response
        st.session_state['response_node'] = response_node

        # Export in json
        response_json = {node: resp for node, resp in response_node.items()}
        st.json(response_json)
        st.download_button("Télécharger les réponses", data=json.dumps(response_json, ensure_ascii=False, indent=2), file_name="responses.json", mime="application/json")
