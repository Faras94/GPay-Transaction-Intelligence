"""
RAG Reranking Module

Reranks retrieved documents using cross-encoder models.
"""

from .rag_models import load_models
from .rag_config import FINAL_K


def rerank_docs(query: str, doc_items: list):
    """
    Rerank documents using cross-encoder.
    
    Args:
        query: User query
        doc_items: List of document dictionaries with 'text' key
        
    Returns:
        list: Reranked documents (top FINAL_K)
    """
    if not doc_items:
        return []
        
    _, reranker = load_models()
    
    # Prepare pairs for reranking
    pairs = [(query, d["text"][:1200]) for d in doc_items]
    rerank_scores = reranker.predict(pairs)
    
    # Attach rerank scores and sort
    for i, d in enumerate(doc_items):
        d["rerank_score"] = float(rerank_scores[i])
        
    # Sort by rerank score descending
    ranked = sorted(doc_items, key=lambda x: x["rerank_score"], reverse=True)
    
    return ranked[:FINAL_K]
