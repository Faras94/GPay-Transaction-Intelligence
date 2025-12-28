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
GLOBAL_DF = None  # Store the full structured dataframe

def initialize_rag(file_path: str = None, df = None, source_file: str = None):
    """Initialize RAG system with PDF file OR DataFrame."""
    global PDF_READY, DOCS, INDEX, EMBEDDING_SOURCE, GLOBAL_DF
    
    try:
        chunks = []
        
        # Priority 1: Use DataFrame if provided (Structured Data)
        if df is not None and not df.empty:
            GLOBAL_DF = df.copy()  # Store for hybrid search
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
            GLOBAL_DF = None # No structured data available if raw PDF used
            
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


def query_structured_data(question: str):
    """
    Attempt to answer using direct DataFrame filtering (Hybrid Search).
    Returns a context string if successful, else None.
    """
    if GLOBAL_DF is None or GLOBAL_DF.empty:
        return None
        
    import re
    import pandas as pd
    
    # 1. Date Extraction (e.g., "29 Nov", "Nov 29", "29/11")
    # Simple regex for finding a date. 
    # Valid GPay formats: "29Nov", "29 Nov", "29 November"
    # Matches: dd mon, dd month
    date_pattern = r'\b(\d{1,2})\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
    match = re.search(date_pattern, question, re.IGNORECASE)
    
    if match:
        day, month_str = match.groups()
        # Normalize Month to Title case (Nov)
        month = month_str[:3].title()
        
        # Filter DF
        # Expected Date column format in GLOBAL_DF is likely datetime objects IF processed, 
        # BUT rags initialize often happens with the raw extracted DF or the processed one.
        # In dashboard.py, we pass 'df' which has "Date" as datetime objects (pd.to_datetime).
        
        try:
             # We need to filter based on Day and Month
             # This assumes GLOBAL_DF['Date'] is datetime. 
             # Let's check safely.
             if not pd.api.types.is_datetime64_any_dtype(GLOBAL_DF['Date']):
                 # Try converting temporary
                 temp_dates = pd.to_datetime(GLOBAL_DF['Date'], errors='coerce')
             else:
                 temp_dates = GLOBAL_DF['Date']
                 
             # Map month name to number
             month_map = {name: i for i, name in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1)}
             target_month_num = month_map.get(month)
             target_day = int(day)
             
             # Apply Filter
             mask = (temp_dates.dt.day == target_day) & (temp_dates.dt.month == target_month_num)
             filtered = GLOBAL_DF[mask]
             
             if not filtered.empty:
                 # Construct specialized context
                 # Limit to reasonable amount (e.g., 50) to avoid token overflow
                 # If > 50, summarize
                 
                 lines = []
                 total_amt = 0
                 
                 for i, row in filtered.iterrows():
                     desc = row.get("Description", "Txn")
                     amt = row.get("Amount (₹)", 0)
                     time = row.get("Time", "")
                     t_type = row.get("Type", "")
                     
                     lines.append(f"- {time}: {desc} | ₹{amt} ({t_type})")
                     total_amt += float(amt) if isinstance(amt, (int, float)) else 0
                 
                 count = len(lines)
                 context_header = f"Found {count} transactions on {day} {month} matching the query."
                 context_body = "\n".join(lines[:60]) # Pass up to 60 transactions
                 if count > 60:
                     context_body += f"\n... (and {count-60} more)"
                     
                 context_summary = f"Total Volume on this day: ₹{total_amt:,.2f}"
                 
                 return f"{context_header}\n{context_summary}\n\nList:\n{context_body}"
                 
        except Exception:
            pass # Fallback to standard vector search if parsing fails
            
    return None


def query_rag(question: str):
    """
    Query the RAG system.
    """
    
    if not PDF_READY or INDEX is None:
        return {
            "answer": "⚠️ RAG system not initialized. Please upload a file first.",
            "sources": []
        }
    
    try:
        # STRATEGY 1: Hybrid Structured Search (Advanced)
        # Check if we can answer directly from the DataFrame (e.g. for specific dates)
        structured_context = query_structured_data(question)
        
        sources = []
        context = ""
        
        if structured_context:
            context = structured_context
            # Expose the actual context in the source for debugging/verification
            sources = [{"type": "structured_db_query", "text": structured_context, "cosine_score": 1.0, "rerank_score": 10.0}]
        else:
            # STRATEGY 2: Standard Vector Search (Fallback)
            # Retrieve documents
            retrieved = retrieve(question, INDEX, DOCS)
            
            if not retrieved:
                return {
                    "answer": "No relevant information found.",
                    "sources": []
                }
            
            # Rerank
            reranked = rerank_docs(question, retrieved)
            sources = reranked
            
            # Build context
            context = "\n\n".join([d["text"] for d in reranked])
        
        # Call LLM
        prompt = f"""You are an expert Financial Analyst Assistant analyzing transaction data.

Context Data:
{context}

User Question: {question}

Instructions:
1. If the context starts with "Found X transactions", this means the data retrieval was SUCCESSFUL. Use this data to answer.
2. Present the information clearly. If a list is provided, you can summarize or present it as requested.
3. Format monetary values as ₹XX.XX.
4. Be direct and helpful.
5. ONLY say "no transactions found" if the context explicitly indicates no data was retrieved.

Answer:"""
        
        answer = call_llm(prompt)
        
        return {
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }
