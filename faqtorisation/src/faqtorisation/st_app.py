import streamlit as st

page_pipe_test = st.Page("pages/st_pipeline_test.py", title="Visualisation des similarités")
page_test = st.Page("pages/st_test.py", title="Test de la visualisation des graphes")


pg = st.navigation([page_pipe_test, page_test])
pg.run()