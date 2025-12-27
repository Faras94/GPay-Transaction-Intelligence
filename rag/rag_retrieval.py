"""
RAG Retrieval Module

Implements hybrid retrieval combining semantic search with keyword matching.
"""

import re
import faiss
from .rag_models import load_models
from .rag_config import TOP_K


def retrieve(query: str, index, docs: list):
    """
    Hybrid retrieval: semantic search + keyword matching.
    
    Args:
        query: User query
        index: FAISS index
        docs: List of document chunks
        
    Returns:
        list: Retrieved documents with scores and types
    """
    embedder, _ = load_models()
    q_emb = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)

    # 1. Semantic Search
    scores, idx = index.search(q_emb, TOP_K)
    
    semantic_results = []
    for i, doc_idx in enumerate(idx[0]):
        if doc_idx < len(docs):
            semantic_results.append({
                "text": docs[doc_idx],
                "cosine_score": float(scores[0][i]),
                "type": "semantic"
            })
            
    # 2. Keyword/Exact Match (Hybrid)
    exact_matches = []
    
    def is_duplicate(doc_text, matches):
        return any(m["text"] == doc_text for m in matches)
    
    # 2a. UPI IDs
    query_upis = re.findall(r'\b\d{12}\b', query)
    if query_upis:
        for doc in docs:
            for upi in query_upis:
                if upi in doc and not is_duplicate(doc, exact_matches):
                    exact_matches.append({
                        "text": doc,
                        "cosine_score": 1.0,
                        "type": "exact_id_match"
                    })
                    break
    
    # 2b. Amounts
    query_amounts = re.findall(r'₹?\s?([\d,]+\.?\d*)', query)
    if query_amounts:
        for amt in query_amounts:
            clean_amt = amt.replace(",", "").replace("₹", "")
            try:
                if float(clean_amt) > 0:
                    for doc in docs:
                        if clean_amt in doc and not is_duplicate(doc, exact_matches):
                            exact_matches.append({
                                "text": doc,
                                "cosine_score": 0.95,
                                "type": "amount_match"
                            })
            except:
                pass

    # 2c. Proper Noun/Keyword Match
    query_keywords = re.findall(r'\b[A-Z][a-zA-Z0-9]*\b', query)
    if query_keywords:
        for doc in docs:
            for kw in query_keywords:
                if len(kw) < 3 or kw.upper() in ["UPI", "OCT", "NOV", "DEC", "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP"]:
                    continue
                    
                if kw in doc and not is_duplicate(doc, exact_matches):
                    exact_matches.append({
                        "text": doc,
                        "cosine_score": 0.9,
                        "type": "keyword_match"
                    })
                    break

    # Combine results
    seen_texts = set()
    combined = []
    
    for item in exact_matches:
        if item["text"] not in seen_texts:
            combined.append(item)
            seen_texts.add(item["text"])
            
    for item in semantic_results:
        if item["text"] not in seen_texts:
            combined.append(item)
            seen_texts.add(item["text"])
            
    return combined[:TOP_K + 3]
