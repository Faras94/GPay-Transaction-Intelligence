"""RAG pipeline validation test.
This test ensures that the core RAG components (retrieval, reranking, LLM) work together.
It runs a single query against a small in‑memory index and checks that the LLM returns a response.
"""
import os
import pytest
from rag.rag_retrieval import RetrievalEngine
from rag.rag_reranking import Reranker
from rag.rag_llm import LLMClient

@pytest.mark.timeout(120)
def test_rag_end_to_end():
    # Ensure environment variable for LLM API key is set (the CI runner will provide it)
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY must be set"

    # 1. Retrieval
    engine = RetrievalEngine()
    query = "What was the total amount spent on coffee last month?"
    docs = engine.search(query, top_k=3)
    assert docs, "Retrieval returned no documents"

    # 2. Reranking
    reranker = Reranker()
    reranked = reranker.rank(query, docs)
    assert reranked, "Reranker returned no documents"

    # 3. LLM generation
    llm = LLMClient()
    prompt = llm.build_prompt(query, reranked)
    answer = llm.generate(prompt)
    assert answer and isinstance(answer, str) and len(answer.strip()) > 0, "LLM returned empty answer"
