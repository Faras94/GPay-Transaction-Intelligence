"""
RAG Package

Modular Retrieval-Augmented Generation system for transaction analysis.

This package provides a loosely coupled RAG implementation with:
- Configurable embeddings and LLM models
- Persistent embedding cache
- Hybrid retrieval (semantic + keyword)
- Cross-encoder reranking
- PDF and DataFrame support
"""

from .rag_pipeline import initialize_rag, query_rag
from .rag_embeddings import file_hash, cache_path

__all__ = [
    'initialize_rag',
    'query_rag',
    'file_hash',
    'cache_path'
]
