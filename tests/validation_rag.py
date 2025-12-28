"""RAG pipeline validation test.
This test verifies the RAG components (retrieval, reranking, LLM) using mocks to avoid external dependencies.
"""
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Import the functions to test
from rag.rag_retrieval import retrieve
from rag.rag_reranking import rerank_docs
from rag.rag_llm import LLMClient

@pytest.fixture
def mock_models():
    """Mock the embedding and reranking models."""
    with patch("rag.rag_retrieval.load_models") as mock_load_retrieval, \
         patch("rag.rag_reranking.load_models") as mock_load_reranking:
        
        # Mock Embedder
        mock_embedder = MagicMock()
        # Return a dummy vector of size (1, 128) for any encode call
        mock_embedder.encode.return_value = np.zeros((1, 128), dtype="float32")
        
        # Mock retrieval load_models returns (embedder, None)
        mock_load_retrieval.return_value = (mock_embedder, None)
        
        # Mock Reranker
        mock_reranker_model = MagicMock()
        # preduct returns a list of scores, one for each pair
        mock_reranker_model.predict.return_value = [0.99, 0.5, 0.1]
        
        # Mock reranking load_models returns (None, reranker)
        mock_load_reranking.return_value = (None, mock_reranker_model)
        
        yield mock_load_retrieval, mock_load_reranking

@pytest.fixture
def dummy_index():
    """Create a dummy FAISS index."""
    import faiss
    index = faiss.IndexFlatL2(128)
    # Add some dummy vectors
    vectors = np.zeros((5, 128), dtype="float32")
    index.add(vectors)
    return index

@pytest.mark.timeout(120)
def test_rag_end_to_end_mocked(mock_models, dummy_index):
    """Test the pipeline flow with mocked models."""
    # Ensure environment variable for LLM API key is set for LLMClient init
    # (Checking strictly, though LLMClient might check it on init)
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "sk-dummy-key"

    query = "Coffee expenses"
    # Docs as strings, as expected by retrieve()
    docs = [
        "Transaction: ₹500 at Coffee Shop",
        "Transaction: ₹100 at Grocery",
        "Transaction: ₹50 at Bus"
    ]

    # 1. Retrieval
    # retrieve signature: retrieve(query: str, index, docs: list)
    # It will call load_models (mocked) loop through docs.
    results = retrieve(query, dummy_index, docs)
    
    # We expect some results. 
    # Since our mock embedder returns zeros and index has zeros, distance is 0.
    # It should match semantically.
    assert len(results) > 0, "Retrieval returned no results"
    assert "text" in results[0]
    
    # 2. Reranking
    # rerank_docs signature: rerank_docs(query, doc_items)
    reranked = rerank_docs(query, results)
    assert len(reranked) > 0, "Reranking returned empty list"
    assert "rerank_score" in reranked[0]

    # 3. LLM Generation
    # We will mock the actual network call to OpenAI in LLMClient
    with patch("rag.rag_llm.LLMClient.generate") as mock_generate:
        mock_generate.return_value = "You spent ₹500 on coffee."
        
        llm = LLMClient()
        prompt = llm.build_prompt(query, reranked)
        answer = llm.generate(prompt)
        
        assert answer == "You spent ₹500 on coffee."
