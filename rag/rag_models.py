"""
RAG Models Module

Handles loading and caching of ML models (embeddings and reranking).
Includes CI-safe mocking to prevent heavy downloads during testing.
"""

import os
import streamlit as st
import numpy as np

# Conditional import to avoid hard dependency if not needed in strictly mocked envs (though usually present)
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    pass  # Handled below if real models are requested but lib missing

from .rag_config import EMBED_MODEL, RERANK_MODEL


class MockEmbedder:
    """Mock SentenceTransformer for CI/CD."""
    def encode(self, sentences, **kwargs):
        # Return dummy vectors of shape (N, 384) typical for MiniLM
        if isinstance(sentences, str):
            sentences = [sentences]
        return np.zeros((len(sentences), 384), dtype="float32")
    
    def get_sentence_embedding_dimension(self):
        return 384


class MockReranker:
    """Mock CrossEncoder for CI/CD."""
    def predict(self, pairs, **kwargs):
        # Return dummy scores typically between 0 and 1
        return [0.95 - (i * 0.1) for i in range(len(pairs))]


@st.cache_resource
def load_models():
    """
    Load embedding and reranking models with caching.
    Returns mocks if os.environ["CI"] is true to avoid downloads.
    """
    if os.getenv("CI") == "true":
        print("⚠️ CI Mode: Loading Mock Models (no HF downloads)")
        return MockEmbedder(), MockReranker()

    # Real initialization
    embedder = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)
    return embedder, reranker
