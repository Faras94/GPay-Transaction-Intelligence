"""
RAG Embeddings Module

Handles embedding generation, caching, and cache management.
"""

import os
import hashlib
import numpy as np
def file_hash(path: str) -> str:
    """Generate MD5 hash of file content."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
