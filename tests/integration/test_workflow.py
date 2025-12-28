
import pytest
from unittest.mock import patch, MagicMock
from rag.rag_pipeline import initialize_rag, query_rag

@patch('rag.rag_pipeline.load_models')
@patch('rag.rag_pipeline.retrieve')
@patch('rag.rag_pipeline.rerank_docs')
@patch('rag.rag_pipeline.call_llm')
def test_full_rag_workflow(mock_llm, mock_rerank, mock_retrieve, mock_load_models, mock_transactions_df):
    """
    Test the full workflow: Initialize -> Query
    """
    import numpy as np
    
    # 1. Setup Mocks
    mock_embedder = MagicMock()
    mock_embedder.get_sentence_embedding_dimension.return_value = 384
    mock_embedder.encode.return_value = np.zeros((2, 384), dtype="float32")
    mock_load_models.return_value = (mock_embedder, None)
    
    mock_retrieve.return_value = [{"text": "doc1", "score": 0.9}]
    mock_rerank.return_value = [{"text": "doc1", "rerank_score": 0.95}]
    mock_llm.return_value = "Integration test answer"
    
    # 2. Initialize
    success, info = initialize_rag(df=mock_transactions_df)
    assert success is True
    assert info["status"] == "generated"
    
    # 3. Query
    # We rely on initialize_rag setting the global state. 
    # Since tests might run in same process, globals persist. This is a known side-effect in this design.
    result = query_rag("Test question")
    
    # 4. Verify
    assert result["answer"] == "Integration test answer"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["text"] == "doc1"
