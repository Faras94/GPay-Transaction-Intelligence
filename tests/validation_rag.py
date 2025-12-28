"""RAG pipeline validation test.
This test ensures that the core RAG components (retrieval, reranking, LLM) work together.
It runs a single query against a small in‑memory index and checks that the LLM returns a response.
"""
import os
import pytest
import numpy as np
import faiss

from rag.rag_retrieval import retrieve
from rag.rag_reranking import rerank_docs
from rag.rag_llm import LLMClient
from rag.rag_document import Document

@pytest.fixture
def dummy_faiss_index_and_docs():
    """Fixture to create a dummy FAISS index and corresponding documents."""
    dimension = 128
    num_docs = 5
    np.random.seed(42)
    vectors = np.random.rand(num_docs, dimension).astype('float32')
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    docs = [
        Document(
            id=f"doc_{i}",
            text=f"This is a dummy document about coffee and expenses. Document number {i}.",
            embedding=vectors[i].tolist()
        )
        for i in range(num_docs)
    ]
    return index, docs, dimension

@pytest.mark.timeout(120)
def test_rag_end_to_end_with_dummy_index(dummy_faiss_index_and_docs):
    # Ensure environment variable for LLM API key is set (the CI runner will provide it)
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY must be set"

    faiss_index, all_docs, embedding_dimension = dummy_faiss_index_and_docs
    query = "What was the total amount spent on coffee last month?"
    query_embedding = np.random.rand(1, embedding_dimension).astype('float32')[0].tolist() # Dummy query

    # 1. Retrieval
    retrieved_docs = retrieve(query_embedding, faiss_index, all_docs, top_k=3)
    assert retrieved_docs, "Retrieval returned no documents"
    assert len(retrieved_docs) <= 3, "Retrieval returned more than top_k documents"
    assert all(isinstance(d, Document) for d in retrieved_docs), "Retrieved items are not Document objects"

    # 2. Reranking
    reranked_docs = rerank_docs(query, retrieved_docs)
    assert reranked_docs, "Reranker returned no documents"
    assert len(reranked_docs) == len(retrieved_docs), "Reranker changed the number of documents"
    assert all(isinstance(d, Document) for d in reranked_docs), "Reranked items are not Document objects"
    # Check if the order might have changed (simple check, not exhaustive)
    assert reranked_docs[0].id != retrieved_docs[0].id or reranked_docs[-1].id != retrieved_docs[-1].id, \
        "Reranker did not change the order of documents, which is unexpected for a functional reranker."

    # 3. LLM generation
    llm = LLMClient()
    prompt = llm.build_prompt(query, reranked_docs)
    answer = llm.generate(prompt)
    assert answer and isinstance(answer, str) and len(answer.strip()) > 0, "LLM returned empty answer"
