"""
RAG Models Module

Handles loading and caching of ML models (embeddings and reranking).
"""

import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from .rag_config import EMBED_MODEL, RERANK_MODEL


@st.cache_resource
def load_models():
    """Load embedding and reranking models with caching."""
    embedder = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)
    return embedder, reranker
