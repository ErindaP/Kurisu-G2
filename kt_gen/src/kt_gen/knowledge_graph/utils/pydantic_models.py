from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Tuple
import networkx as nx
import numpy as np

# --- Pydantic Models ---

class Node(BaseModel):
    index: str
    text: str
    type: str = Field(default="unknown")


class TraversalStep(BaseModel):
    node_id: str
    similarity: Optional[float]  # None pour la racine


class EmbeddingStore(BaseModel):
    vectors: Dict[str, np.ndarray]
    model_config = ConfigDict(arbitrary_types_allowed=True)



class GraphWrapper(BaseModel):
    graph: nx.Graph

    class Config:
        arbitrary_types_allowed = True


class EmbeddingMatrix(BaseModel):
    vectors: np.ndarray

    class Config:
        arbitrary_types_allowed = True


