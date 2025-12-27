"""
RAG Processing Module

Handles PDF reading, text chunking, and DataFrame processing.
"""

import re
import fitz


def read_pdf(file_path: str) -> str:
    """Read text from PDF file."""
    doc = fitz.open(file_path)
    return "\n".join(page.get_text() for page in doc)


def chunk_text(text: str, size=800, overlap=100) -> list:
    """
    Chunk text into overlapping segments.
    
    Args:
        text: Input text
        size: Chunk size in characters
        overlap: Overlap between chunks
        
    Returns:
        list: Text chunks
    """
    chunks = []
    step = size - overlap
    for i in range(0, len(text), step):
        chunk = text[i:i + size].strip()
        if len(chunk) > 200:
            chunks.append(chunk)
    return chunks


def dataframe_to_chunks(df) -> list:
    """
    Convert DataFrame rows to text chunks for indexing.
    
    Args:
        df: Transaction DataFrame
        
    Returns:
        list: Text chunks representing transactions
    """
    chunks = []
    for _, row in df.iterrows():
        chunk = f"""
Date: {row.get('Date', 'N/A')}
Time: {row.get('Time', 'N/A')}
Description: {row.get('Description', 'N/A')}
Amount: ₹{row.get('Amount (₹)', 0)}
Type: {row.get('Type', 'N/A')}
Category: {row.get('Category', 'N/A')}
UPI ID: {row.get('UPI ID', 'N/A')}
""".strip()
        chunks.append(chunk)
    return chunks


def transaction_logic(question: str, context: str):
    """
    Domain-specific logic for transaction queries.
    
    Args:
        question: User question
        context: Retrieved context
        
    Returns:
        str: Direct answer if pattern matches, else None
    """
    q = question.lower()

    if "first transaction" in q:
        dates = re.findall(r"\d{1,2}\s+\w+,\s+\d{4}", context)
        if dates:
            return f"The first transaction occurred on {dates[0]}."

    amt = re.search(r"₹\s?([\d,]+)", question)
    if amt:
        value = amt.group(1).replace(",", "")
        lines = context.splitlines()
        received = paid = None

        for i, line in enumerate(lines):
            if value in line.replace(",", ""):
                block = "\n".join(lines[i:i + 6])
                if "Received from" in block and not received:
                    received = block.split("Received from")[-1].split("\n")[0].strip()
                if "Paid to" in block and not paid:
                    paid = block.split("Paid to")[-1].split("\n")[0].strip()

        if received and paid:
            return f"₹{amt.group(1)} was received from {received} and later paid to {paid}."

    return None
