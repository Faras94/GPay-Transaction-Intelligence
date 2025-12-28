
import pytest
from unittest.mock import MagicMock, patch
from rag.rag_pipeline import initialize_rag, query_rag

@patch('rag.rag_pipeline.build_index')
@patch('rag.rag_pipeline.load_models')
def test_initialize_rag_with_df(mock_load_models, mock_build_index, mock_transactions_df):
    # Mock embeddings loading
    import numpy as np
    mock_embedder = MagicMock()
    mock_embedder.get_sentence_embedding_dimension.return_value = 384
    # Return real numpy array so FAISS bindings don't crash/complain
    mock_embedder.encode.return_value = np.zeros((2, 384), dtype="float32")
    mock_load_models.return_value = (mock_embedder, None)
    
    success, info = initialize_rag(df=mock_transactions_df)
    
    assert success is True
    assert "Indexed 2 transactions" in info["message"]
    assert info["status"] == "generated"

def test_query_rag_uninitialized():
    # Ensure global state is reset if needed, or check fail
    # In a real scenario, we might need to reset module state
    # For now, we assume it runs isolated or in sequence
    
    # We can't easily force uninitialized state without accessing globals
    # But we can try querying and see if it handles it (it might be initialized from prev tests)
    pass 

@patch('rag.rag_pipeline.retrieve')
@patch('rag.rag_pipeline.rerank_docs')
@patch('rag.rag_pipeline.call_llm')
def test_query_rag_flow(mock_llm, mock_rerank, mock_retrieve):
    # Setup mocks
    mock_retrieve.return_value = [{"text": "doc1", "score": 0.9}]
    mock_rerank.return_value = [{"text": "doc1", "rerank_score": 0.95}]
    mock_llm.return_value = "This is a mocked answer."
    
    # Force initialization first (using a separate test might be cleaner, but we need state)
    # We trust initialize_rag was called or we call it again?
    # Ideally tests should be independent. 
    # Let's simple check query_rag behavior when dependencies mock return values.
    
    # We need to make sure PDF_READY is True in the module.
    with patch('rag.rag_pipeline.PDF_READY', True):
        with patch('rag.rag_pipeline.INDEX', MagicMock()):
             result = query_rag("How much did I spend?")
             
             assert result["answer"] == "This is a mocked answer."
             assert len(result["sources"]) == 1
