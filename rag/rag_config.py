"""
RAG Configuration Module

Centralized configuration for the RAG system including API keys,
model names, and system constants.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ================= API CONFIGURATION =================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ================= MODEL CONFIGURATION =================
LLM_MODEL = "allenai/olmo-3.1-32b-think:free"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ================= RETRIEVAL PARAMETERS =================
TOP_K = 25  # Increased to capture more candidates
FINAL_K = 15  # Increased to allow longer lists in answers

# ================= CACHE CONFIGURATION =================
CACHE_DIR = "embed_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
