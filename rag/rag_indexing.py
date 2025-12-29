"""
RAG Indexing Module

Handles FAISS index creation and management.
"""

import faiss
import numpy as np
from .rag_models import load_models
from .rag_embeddings import file_hash


def build_index(file_path: str, chunks: list):
    """
    Build FAISS index by generating fresh embeddings.
    
    Args:
        file_path: Path to source file (for hash)
        chunks: List of text chunks to index
        
    Returns:
        dict: Status information with index and docs
    """
    embedder, _ = load_models()
    local_dim = embedder.get_sentence_embedding_dimension()
    h = file_hash(file_path)
    
    # Generate fresh embeddings
    embeddings = embedder.encode(chunks).astype("float32")
    faiss.normalize_L2(embeddings)
    
    index = faiss.IndexFlatIP(local_dim)
    index.add(embeddings)
    
    return {
        "index": index,
        "docs": chunks,
        "status": "generated",
        "chunk_count": len(chunks),
        "source": "Generated (no cache)",
        "file_hash": h
    }
