import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Laden eines Pre-Trained Sentence Transformer-Modells
sent_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def create_embeddings(text_list):
    """
    Erzeugt Vektor-Embeddings für eine Liste von Texten.
    """
    embeddings = sent_model.encode(text_list, convert_to_tensor=False)
    return np.array(embeddings)

def create_index(embeddings):
    """
    Erstellt einen FAISS Index basierend auf den Embeddings.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_index(index, query_embedding, k=5):
    """
    Sucht im Index nach den k ähnlichsten Einträgen.
    """
    distances, indices = index.search(np.array([query_embedding]), k)
    return indices[0]
