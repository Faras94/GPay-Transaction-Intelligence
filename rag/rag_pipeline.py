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
            
            # Simple, non-cached indexing
            result = build_index(source_file if source_file else "df_source", chunks)
            INDEX = result["index"]
            DOCS = result["docs"]
            EMBEDDING_SOURCE = "dataframe"
            PDF_READY = True
            
            result["message"] = f"✅ Indexed {result['chunk_count']} transaction chunks"
            return True, result
            
        # Priority 2: Use PDF file if provided (Unstructured Data)
        elif file_path and os.path.exists(file_path):
            text = read_pdf(file_path)
            chunks = chunk_text(text)
            
            if not chunks:
                return False, {"message": "❌ No text chunks extracted from PDF"}
            
            result = build_index(file_path, chunks)
            INDEX = result["index"]
            DOCS = result["docs"]
            EMBEDDING_SOURCE = "generated"
            PDF_READY = True
            GLOBAL_DF = None # No structured data available if raw PDF used
            
            result["message"] = f"✅ Generated {result['chunk_count']} chunks"
            return True, result
        else:
            return False, {"message": "❌ No data provided for indexing"}
        
    except Exception as e:
        import traceback
        return False, {"message": f"❌ Error: {str(e)}\n{traceback.format_exc()}"}


def _generate_analytics_context(filtered, context_header):
    """
    Helper to generate advanced analytics context from a filtered DataFrame.
    """
    import pandas as pd
    # 1. Exact Totals
    total_spent = filtered[filtered['Type'] == 'Spent']['Amount (₹)'].sum()
    total_received = filtered[filtered['Type'] == 'Received']['Amount (₹)'].sum()
    txn_count = len(filtered)
    
    # 2. Category Breakdown
    cat_summary = ""
    if 'Category' in filtered.columns:
        cat_stats = filtered[filtered['Type'] == 'Spent'].groupby('Category')['Amount (₹)'].sum().sort_values(ascending=False)
        cat_list = []
        for cat, amt in cat_stats.head(5).items():
            cat_list.append(f"- {cat}: ₹{amt:,.2f}")
        if not cat_list:
            cat_list.append("- No category data available")
        cat_summary = "Top Categories (Spending):\n" + "\n".join(cat_list)

    # 3. Smart Listing (Up to 200 items)
    lines = []
    limit = 200 
    
    sorted_filtered = filtered.sort_values(by='Date', ascending=False)
    
    for i, row in sorted_filtered.iterrows():
        desc = row.get("Description", "Txn")
        if len(desc) > 30: desc = desc[:28] + ".."
        
        amt = row.get("Amount (₹)", 0)
        date = row.get("Date").strftime("%d %b %Y") if pd.notnull(row.get("Date")) else ""
        cat = row.get("Category", "-")
        t_type = row.get("Type", "Spent")
        type_icon = "🔴" if t_type == "Spent" else "🟢"
        
        lines.append(f"| {date} | {desc} | {type_icon} ₹{amt:,.2f} | {cat} |")
    
    display_count = len(lines)
    truncated = False
    if display_count > limit:
        lines = lines[:limit]
        truncated = True
    
    context_body = "\n".join(lines)
    
    context_analytics = f"""
=== SUMMARY STATISTICS (Calculated from full data) ===
Total Spent: ₹{total_spent:,.2f}
Total Received: ₹{total_received:,.2f}
Net Flow: ₹{(total_received - total_spent):,.2f}
Transaction Count: {txn_count}

{cat_summary}
"""
    table_header = "| Date | Description | Type/Amount | Category |\n|---|---|---|---|"
    
    full_context = f"{context_header}\n{context_analytics}\n\n=== TRANSACTION LIST ({len(lines)} shown) ===\n{table_header}\n{context_body}"
    
    if truncated:
        full_context += f"\n\n... (and {txn_count - limit} more transactions accounted for in Summary)"
    
    return full_context


def query_structured_data(question: str):
    """
    Attempt to answer using direct DataFrame filtering (Hybrid Search).
    Returns a context string if successful, else None.
    """
    global GLOBAL_DF
    if GLOBAL_DF is None or GLOBAL_DF.empty:
        return None
        
    import re
    import pandas as pd
    
    # 1. Day Month Extraction (e.g. "22 Oct")
    # Matches: dd mon, dd month
    date_pattern = r'\b(\d{1,2})\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
    match_day = re.search(date_pattern, question, re.IGNORECASE)

    if match_day:
        day, month_str = match_day.groups()
        month = month_str[:3].title()
        
        try:
             if not pd.api.types.is_datetime64_any_dtype(GLOBAL_DF['Date']):
                 temp_dates = pd.to_datetime(GLOBAL_DF['Date'], errors='coerce')
             else:
                 temp_dates = GLOBAL_DF['Date']
                 
             month_map = {name: i for i, name in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1)}
             target_month_num = month_map.get(month)
             target_day = int(day)
             
             # Apply Filter
             mask = (temp_dates.dt.day == target_day) & (temp_dates.dt.month == target_month_num)
             filtered = GLOBAL_DF[mask]
             
             if not filtered.empty:
                 return _generate_analytics_context(filtered, f"Found {len(filtered)} transactions on {day} {month} matching the query.")
        except Exception:
            pass

    # 2. Month Year Extraction (e.g. "October 2025", "Oct 2025")
    month_year_pattern = r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*,?\s*(\d{4})\b'
    match_my = re.search(month_year_pattern, question, re.IGNORECASE)
    
    if match_my:
        month_str, year_str = match_my.groups()
        month = month_str[:3].title()
        
        try:
             if not pd.api.types.is_datetime64_any_dtype(GLOBAL_DF['Date']):
                 temp_dates = pd.to_datetime(GLOBAL_DF['Date'], errors='coerce')
             else:
                 temp_dates = GLOBAL_DF['Date']
                 
             month_map = {name: i for i, name in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1)}
             target_month_num = month_map.get(month)
             target_year = int(year_str)
             
             # Apply Filter
             mask = (temp_dates.dt.month == target_month_num) & (temp_dates.dt.year == target_year)
             filtered = GLOBAL_DF[mask]
             
             if not filtered.empty:
                 return _generate_analytics_context(filtered, f"Found {len(filtered)} transactions in {month} {year_str}.")
                 
        except Exception:
            pass

    # 3. Month Only Extraction (e.g. "October", "in Oct")
    # This runs ONLY if the specific Day/Year patterns above didn't trigger/return.
    month_only_pattern = r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b'
    match_mon = re.search(month_only_pattern, question, re.IGNORECASE)
    
    if match_mon:
        month_str = match_mon.group(1)
        month = month_str[:3].title()
        
        try:
             if not pd.api.types.is_datetime64_any_dtype(GLOBAL_DF['Date']):
                 temp_dates = pd.to_datetime(GLOBAL_DF['Date'], errors='coerce')
             else:
                 temp_dates = GLOBAL_DF['Date']
                 
             month_map = {name: i for i, name in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1)}
             target_month_num = month_map.get(month)
             
             # Apply Filter (All Years)
             mask = (temp_dates.dt.month == target_month_num)
             filtered = GLOBAL_DF[mask]
             
             if not filtered.empty:
                 return _generate_analytics_context(filtered, f"Found {len(filtered)} transactions in {month} (All Years).")

        except Exception:
            pass

    # 4. Category/Keyword Extraction
    # Check if any known category is in the query
    if 'Category' in GLOBAL_DF.columns:
        unique_cats = GLOBAL_DF['Category'].dropna().unique()
        for cat in unique_cats:
            # Simple keyword match
            if cat.lower() in question.lower():
                filtered = GLOBAL_DF[GLOBAL_DF['Category'] == cat]
                if not filtered.empty:
                    return _generate_analytics_context(filtered, f"Found {len(filtered)} transactions for category '{cat}'.")

    # 5. Generic Total/All Time
    # If users ask "total spent" or "all transactions" without date filters
    if any(k in question.lower() for k in ["total spent", "how much spent", "all transactions", "overall", "all time"]):
        return _generate_analytics_context(GLOBAL_DF, f"Found {len(GLOBAL_DF)} transactions (All Time).")

    return None


def query_rag(question: str):
    """
    Query the RAG system.
    """
    global PDF_READY, DOCS, INDEX
    
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
1. If the context contains a structured table or list of transactions, PRESERVE that structure in your response efficiently.
2. If "Found X transactions" is in the context, use that data as the source of truth.
3. When listing transactions, ALWAYS use a Markdown table format:
   | Date | Description | Amount | Category |
   |---|---|---|---|
4. Format monetary values as ₹XX.XX.
5. Provide a summary of the total spending if multiple items are listed.
6. Be direct and helpful.
7. ONLY say "no transactions found" if the context explicitly indicates no data was retrieved.

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
