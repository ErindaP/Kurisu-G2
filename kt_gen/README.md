# Kurisu-G²

## Partie Documentation/Téléchargement

## Doc Kt_Gen 

## Installation

Il y un poetry pour installer les dépendances du projet. 

```bash
poetry install

```
puis
```bash
poetry env activate
```

## Lancement
J'ai pas fait les scripts de lancement par poetry, les pages streamlits principales permettant l'accès aux fonctionnalités sont regroupées dans l'app `st_app.py` et sont accessibles en lançant le script suivant :

```bash
streamlit run src/kt_gen/st_app.py
```

## Organisation du projet

Le projet est organisé de la manière suivante :
```
├── src
│   ├── kt_gen
│   │   ├── __init__.py
│   │   ├── st_app.py
│   │   ├── utils
│   │   │   ├── __init__.py
│   │   │   ├── llm
│   │   │   ├── streamlit
│   │   │   ├── model
│   │   ├── pages
│   │   ├── knowledge_graph
│   │   │   ├── utils
│   │   ├── examples
```

Les scripts conernant les graphes de connaissances sont dans le dossier `knowledge_graph`, les scripts utilitaires sont dans le dossier `utils` et les pages streamlit sont dans le dossier `pages`.
Toutes les versions utilisant la FGW sont dans le fichier `kg_fgw.py` et les versions de base sont dans le fichier `kg_base_algo.py`.

Les exemples de graphes de connaissances sont dans le dossier `examples`.

Le `.env` est utilisé pour le chemin d'accès au modèle d'embedding choisi, par défaut c'est le modèle `all-MiniLM-L6-v2` de HuggingFace qui sera installé depuis le web si rien n'est précisé dans le `.env`.

L'addresse du localhost de ollama est hardcodée dans le code et est l'addresse basique sur le port 11434, je n'ai pas encore pu faire la transformation avec le .env.

## Pages Streamlit

Les pages streamlit sont organisées de la manière suivante :
```
├── pages
│   ├── kg_text_brut.py
│   ├── pipe_IA.py
│   ├── st_graph_llm.py
│   ├── st_pipeline_tests.py
│   ├── visu_graph.py

```

la page `kg_text_brut.py` permet de créer un graphe de connaissance à partir d'un texte brut, les pages `st_graph_llm.py` et `st_pipeline_tests.py` permettent de tester les fonctionnalités des différentes méthodes sur les graphes en posant des questions puis de sauvegarder les graphes de connaissances générés, la page `visu_graph.py` permet de visualiser les graphes de connaissances générés et la page `pipe_IA.py` permet de tester les pipelines d'IA sur les graphes de connaissances.




