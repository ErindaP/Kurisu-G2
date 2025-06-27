import os
from openai import OpenAI

def build_prompt_path(question, path, G):
    context = "\n".join([f"{nid}: {G.nodes[nid]['text']}" for nid in path])
    return f"""Voici les extraits traversés dans le graphe :

{context}

Question : {question}
Réponse :"""

def build_prompt(question, context):
    return f"""Voici des extraits de documents pertinents  :
{context}

Question : {question}
Réponse :"""

def ask_llm(prompt, client):
    response = client.chat.completions.create(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content




# Fonction de résumé du contexte avec axe principaux. 

def resume_context(context,client):
    prompt= f""" Voici des concaténations de phrases extraites : 
{context}
Effectue un résumé de ces élements en extrayant les axes principaux et les idées clés.
Résumé :"""
    response = ask_llm(prompt, client)
    return response


def summurize_axe(context, client):
    prompt = f"""Voici des extraits de documents pertinents :
{context}
Fais une liste de tous les axes principaux qui sont explicitement mentionnés dans ces extraits.
Liste des axes principaux :"""
    response = ask_llm(prompt, client)
    return response

