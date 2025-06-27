import streamlit as st
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import json
import os



# Traitement du texte
def structure_text(text, nlp):
    graph = nx.DiGraph()
    doc_id = "document"
    graph.add_node(doc_id, type="document", text="Document")

    sections = text.split("\n\n")
    section_count = 0

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        section_id = f"section_{section_count}"
        graph.add_node(section_id, type="section", text=sec.split("\n\n")[0])
        graph.add_edge(doc_id, section_id)

        paragraphs = sec.split("\n")
        for p_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            para_id = f"{section_id}_para_{p_idx}"
            graph.add_node(para_id, type="paragraph", text=paragraph)
            graph.add_edge(section_id, para_id)

            doc = nlp(paragraph)
            if ":" in paragraph:
                sent_id = f"{para_id}_sent_0"
                graph.add_node(sent_id, type="sentence", text=paragraph)
                graph.add_edge(para_id, sent_id)
            else:
                for s_idx, sent in enumerate(doc.sents):
                    sent_id = f"{para_id}_sent_{s_idx}"
                    graph.add_node(sent_id, type="sentence", text=sent.text)
                    graph.add_edge(para_id, sent_id)

        section_count += 1

    return graph