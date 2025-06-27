from sentence_transformers import SentenceTransformer
import requests
from huggingface_hub import configure_http_backend
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

modelPath = "./utils/sentence_transformer_model"

model = SentenceTransformer('all-MiniLM-L6-v2')
model.save(modelPath)
