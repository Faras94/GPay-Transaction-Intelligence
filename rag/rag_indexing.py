"""
RAG Indexing Module

Handles FAISS index creation and management.
"""

import faiss
import numpy as np
from .rag_models import load_models
from .rag_embeddings import file_hash, load_cached_embeddings, save_embeddings_to_cache


def build_index(file_path: str, chunks: list):
    """
    Build FAISS index using cached or fresh embeddings.
    
    Args:
        file_path: Path to source file (for cache key)
        chunks: List of text chunks to index
        
    Returns:
        dict: Status information with index, docs, and cache details
    """
    embedder, _ = load_models()
    local_dim = embedder.get_sentence_embedding_dimension()
    h = file_hash(file_path)
    
    # Try loading from cache
    cached_embeddings, cached_docs = load_cached_embeddings(h)
    
    if cached_embeddings is not None and cached_embeddings.shape[1] == local_dim:
        # Use cached embeddings
        faiss.normalize_L2(cached_embeddings)
        index = faiss.IndexFlatIP(local_dim)
        index.add(cached_embeddings)
        
        return {
            "index": index,
            "docs": cached_docs,
            "status": "cache_hit",
            "chunk_count": len(cached_docs),
            "source": "Loaded from cache",
            "file_hash": h
        }
    
    # Generate fresh embeddings
    embeddings = embedder.encode(chunks).astype("float32")
    faiss.normalize_L2(embeddings)
    
    index = faiss.IndexFlatIP(local_dim)
    index.add(embeddings)
    
    # Save to cache
    save_embeddings_to_cache(h, embeddings, chunks)
    
    return {
        "index": index,
        "docs": chunks,
        "status": "generated",
        "chunk_count": len(chunks),
        "source": "Generated and cached",
        "file_hash": h
    }
