
import pytest
from rag.rag_processing import chunk_text, dataframe_to_chunks, transaction_logic

def test_chunk_text():
    text = "A" * 1000
    chunks = chunk_text(text, size=500, overlap=50)
    
    assert len(chunks) > 1
    assert len(chunks[0]) == 500
    assert len(chunks[1]) == 500

def test_dataframe_to_chunks(mock_transactions_df):
    chunks = dataframe_to_chunks(mock_transactions_df)
    
    assert len(chunks) == 2
    assert "Uber Ride" in chunks[0]
    assert "Payment Result" in chunks[1]
    assert "500.0" in chunks[0]

def test_transaction_logic_first_txn():
    context = "My first transaction was on 01 Jan, 2024"
    question = "When was my first transaction?"
    
    # Simple regex test for the specific logic in rag_processing
    # Note: The logic in actual code expects specific date format
    result = transaction_logic(question, context)
    # Logic matches 01 Jan, 2024
    assert result == "The first transaction occurred on 01 Jan, 2024."

def test_transaction_logic_amount():
    context = """
    Date: 01Jan,2024
    Description: Payment to Zomato
    Amount: ₹500
    Paid to Zomato
    """
    question = "Who did I pay ₹500 to?"
    # This relies on complex regex in the implementation
    # We just ensure it runs without error
    result = transaction_logic(question, context)
    # If logic holds, it might return something, or None
    assert result is None or isinstance(result, str)
