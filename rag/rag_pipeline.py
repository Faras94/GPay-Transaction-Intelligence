"""
RAG Pipeline Module

Main orchestration layer for the RAG system.
Manages state and coordinates all other modules.
"""

import os
import faiss
import numpy as np
from .rag_models import load_models
from .rag_indexing import build_index
from .rag_processing import read_pdf, chunk_text, dataframe_to_chunks, transaction_logic
from .rag_retrieval import retrieve
from .rag_reranking import rerank_docs
from .rag_llm import call_llm

# ================= STATE =================
DOCS = []
INDEX = None
PDF_READY = False
EMBEDDING_SOURCE = "unknown"


def initialize_rag(file_path: str = None, df = None, source_file: str = None):
    """
    Initialize RAG system with PDF file OR DataFrame.
    
    Args:
        file_path: Direct PDF path (for unstructured indexing)
        df: DataFrame with transaction data (for structured indexing)
        source_file: Original source file path (used for caching when df is provided)
        
    Returns:
        tuple: (success: bool, status_info: dict)
    """
    global PDF_READY, DOCS, INDEX, EMBEDDING_SOURCE
    
    try:
        chunks = []
        
        # Priority 1: Use DataFrame if provided (Structured Data)
        if df is not None and not df.empty:
            chunks = dataframe_to_chunks(df)
            
            if not chunks:
                return False, {"message": "❌ No chunks generated from DataFrame"}
            
            # Use caching if source_file is provided
            if source_file and os.path.exists(source_file):
                result = build_index(source_file, chunks)
                INDEX = result["index"]
                DOCS = result["docs"]
                EMBEDDING_SOURCE = result["status"]
                PDF_READY = True
                
                # Format message
                if result["status"] == "cache_hit":
                    result["message"] = f"⚡ Loaded {result['chunk_count']} transaction chunks from cache"
                else:
                    result["message"] = f"✅ Generated and cached {result['chunk_count']} transaction chunks"
                
                return True, result
            else:
                # Fallback: Build in-memory index without caching
                embedder, _ = load_models()
                local_dim = embedder.get_sentence_embedding_dimension()
                
                DOCS = chunks
                embeddings = embedder.encode(DOCS).astype("float32")
                faiss.normalize_L2(embeddings)

                INDEX = faiss.IndexFlatIP(local_dim)
                INDEX.add(embeddings)
                EMBEDDING_SOURCE = "dataframe"
                
                PDF_READY = True
                return True, {
                    "status": "generated",
                    "chunk_count": len(DOCS),
                    "source": "DataFrame (in-memory, no cache)",
                    "message": f"✅ Indexed {len(DOCS)} transactions (no cache)"
                }
            
        # Priority 2: Use PDF file if provided (Unstructured Data)
        elif file_path and os.path.exists(file_path):
            text = read_pdf(file_path)
            chunks = chunk_text(text)
            
            if not chunks:
                return False, {"message": "❌ No text chunks extracted from PDF"}
            
            result = build_index(file_path, chunks)
            INDEX = result["index"]
            DOCS = result["docs"]
            EMBEDDING_SOURCE = result["status"]
            PDF_READY = True
            
            # Format message
            if result["status"] == "cache_hit":
                result["message"] = f"⚡ Loaded {result['chunk_count']} chunks from cache"
            else:
                result["message"] = f"✅ Generated and cached {result['chunk_count']} chunks"
            
            return True, result
        else:
            return False, {"message": "❌ No data provided for indexing"}
        
    except Exception as e:
        import traceback
        return False, {"message": f"❌ Error: {str(e)}\n{traceback.format_exc()}"}


def query_rag(question: str):
    """
    Query the RAG system.
    
    Args:
        question: User question
        
    Returns:
        dict: {'answer': str, 'sources': list}
    """
    global PDF_READY, DOCS, INDEX
    
    if not PDF_READY or INDEX is None:
        return {
            "answer": "⚠️ RAG system not initialized. Please upload a file first.",
            "sources": []
        }
    
    try:
        # Retrieve documents
        retrieved = retrieve(question, INDEX, DOCS)
        
        if not retrieved:
            return {
                "answer": "No relevant information found.",
                "sources": []
            }
        
        # Rerank
        reranked = rerank_docs(question, retrieved)
        
        # Build context
        context = "\n\n".join([d["text"] for d in reranked])
        
        # Try domain logic first
        direct_answer = transaction_logic(question, context)
        if direct_answer:
            return {
                "answer": direct_answer,
                "sources": reranked
            }
        
        # Call LLM
        prompt = f"""You are a helpful assistant analyzing transaction data.

Context:
{context}

Question: {question}

Provide a clear, concise answer based on the context above."""
        
        answer = call_llm(prompt)
        
        return {
            "answer": answer,
            "sources": reranked
        }
        
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }
