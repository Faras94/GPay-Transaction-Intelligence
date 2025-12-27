"""
RAG Embeddings Module

Handles embedding generation, caching, and cache management.
"""

import os
import hashlib
import numpy as np
from .rag_config import CACHE_DIR


def file_hash(path: str) -> str:
    """Generate MD5 hash of file content."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def cache_path(hash_id: str) -> str:
    """Get cache file path for a given hash ID."""
    return os.path.join(CACHE_DIR, f"pdf_{hash_id}.npz")


def load_cached_embeddings(hash_id: str):
    """
    Load embeddings from cache if available.
    
    Returns:
        tuple: (embeddings, docs) if cache exists, else (None, None)
    """
    path = cache_path(hash_id)
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        return data["embeddings"].astype("float32"), list(data["docs"])
    return None, None


def save_embeddings_to_cache(hash_id: str, embeddings: np.ndarray, docs: list):
    """Save embeddings and documents to cache."""
    path = cache_path(hash_id)
    np.savez_compressed(path, embeddings=embeddings, docs=np.array(docs, dtype=object))
