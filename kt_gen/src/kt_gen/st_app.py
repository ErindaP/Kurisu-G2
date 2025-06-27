import streamlit as st



#st.set_page_config(page_title="Graph LLM", page_icon=":bar_chart:", layout="wide")

page_graph_llm = st.Page("pages/st_graph_llm.py", title="Graph LLM")
page_kg_text_brut = st.Page("pages/kg_text_brut.py", title="KG Text Brut")
page_pipline_tests = st.Page("pages/st_pipeline_tests.py", title="Pipeline Tests")
pipe_ia = st.Page("pages/pipe_IA.py", title="Pipeline IA")
visu_graph = st.Page("pages/visu_graph.py", title="Visualisation Graph")
pg = st.navigation([page_graph_llm, page_kg_text_brut, page_pipline_tests, pipe_ia, visu_graph])
pg.run()
"""
def main():
    st.set_page_config(
    page_title="Knowledge Graph",
    page_icon=":bar_chart:",
    layout="wide"
    )
    st.write("Sélectionner une page à gauche pour démarrer.")
    script_path = os.path.abspath(__file__)
    subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])


"""