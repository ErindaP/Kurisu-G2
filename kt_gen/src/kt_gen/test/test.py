import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# Données en 2D (x, y) + infos associées
points = [
    {"id": "P1", "x": 1, "y": 2, "info": "Point 1 : Coordonnées (1,2)"},
    {"id": "P2", "x": 3, "y": 1, "info": "Point 2 : Coordonnées (3,1)"},
    {"id": "P3", "x": 2, "y": 4, "info": "Point 3 : Coordonnées (2,4)"},
]

# Préparation des listes
x_vals = [p["x"] for p in points]
y_vals = [p["y"] for p in points]
labels = [p["id"] for p in points]
infos = [p["info"] for p in points]

# Création du scatter plot interactif
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x_vals,
    y=y_vals,
    mode='markers+text',
    text=labels,
    textposition='top center',
    marker=dict(size=10, color='blue'),
    hovertext=infos,
    hoverinfo="text"
))

fig.update_layout(
    title="Cliquez sur un point pour afficher l'information",
    xaxis_title="X",
    yaxis_title="Y"
)

# Affichage avec capture des clics
selected_points = plotly_events(fig, click_event=True, hover_event=False)

# Affichage de l'info du point cliqué
if selected_points:
    point_id = selected_points[0]['pointIndex']
    point_info = points[point_id]['info']
    st.write(f"Information du point sélectionné : {point_info}")
