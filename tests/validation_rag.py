"""RAG pipeline validation test.
This test verifies the RAG components (retrieval, reranking, LLM) using mocks to avoid external dependencies.
"""
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Import the functional API
from rag.rag_retrieval import retrieve
from rag.rag_reranking import rerank_docs
from rag.rag_llm import call_llm

@pytest.fixture
def mock_models():
    """Mock the embedding and reranking models."""
    with patch("rag.rag_retrieval.load_models") as mock_load_retrieval, \
         patch("rag.rag_reranking.load_models") as mock_load_reranking:
        
        # Mock Embedder
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.zeros((1, 128), dtype="float32")
        # Mock retrieval load_models returns (embedder, None)
        mock_load_retrieval.return_value = (mock_embedder, None)
        
        # Mock Reranker
        mock_reranker_model = MagicMock()
        # predict returns a list of scores, one for each pair
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
    # Ensure environment variable for LLM API key handling
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
    results = retrieve(query, dummy_index, docs)
    assert len(results) > 0, "Retrieval returned no results"
    
    # 2. Reranking
    reranked = rerank_docs(query, results)
    assert len(reranked) > 0, "Reranking returned empty list"

    # 3. LLM Generation
    # Mock call_llm internal request locally
    with patch("rag.rag_llm.requests.post") as mock_post:
        # Mock a successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "You spent ₹500 on coffee."}}]
        }
        mock_post.return_value = mock_response
        
        # Since logic constructs prompt manually in pipeline, we just pass a string here
        prompt = f"Context: {reranked[0]['text']}\nQuestion: {query}"
        answer = call_llm(prompt)
        
        assert answer == "You spent ₹500 on coffee."
