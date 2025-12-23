import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np


def create_embeddings(docs, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Generate embeddings for a list of documents."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, convert_to_numpy=True, show_progress_bar=True)
    return embeddings


def save_faiss_index(embeddings, metadata, index_path, metadata_path):
    """Save FAISS index and metadata to disk."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Saved FAISS index to {index_path}")
    print(f"✅ Saved metadata to {metadata_path}")


def load_faiss_index(index_path, metadata_path):
    """Load FAISS index and metadata."""
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata
